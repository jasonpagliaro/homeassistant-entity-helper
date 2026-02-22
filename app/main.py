from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import delete, func, or_
from sqlmodel import Session, select
from starlette.middleware.sessions import SessionMiddleware

from app.automation_drafts import (
    TEMPLATE_CATALOG,
    build_automation_structured_payload,
    build_draft_prompt_payload,
    draft_limit_from_env,
    pick_template_id,
    render_automation_yaml,
)
from app.db import get_engine, get_session, run_migrations
from app.ha_client import HAClient, HAClientError
from app.llm_client import (
    LLMClientError,
    LLMSettings,
    OpenAICompatibleLLMClient,
    llm_is_configured,
    load_llm_settings,
)
from app.models import (
    AutomationDraft,
    AutomationDraftRun,
    ConfigSnapshot,
    ConfigSyncRun,
    EntitySnapshot,
    EntitySuggestion,
    EntitySuggestionRun,
    LLMConnection,
    Profile,
    SuggestionAuditEvent,
    SuggestionProposal,
    SuggestionRun,
    SyncRun,
    utcnow,
)
from app.suggestion_context import build_automation_context, build_known_entity_ids
from app.suggestion_schema import validate_suggestion_payload
from app.suggestion_worker import SuggestionWorker
from app.suggestions import (
    ACTIONABLE_SUGGESTION_DOMAINS,
    SUGGESTION_POLICY_VERSION,
    WORKFLOW_FIXABLE_ISSUE_CODES,
    build_llm_suggestion_payload,
    evaluate_entity_snapshot,
    is_supported_entity,
    manual_guidance_for_issue,
    split_issues_by_fixability,
)

logger = logging.getLogger("ha_entity_vault")
CSRF_SESSION_KEY = "csrf_token"
FLASH_SESSION_KEY = "flash"
ACTIVE_PROFILE_SESSION_KEY = "active_profile_id"
DEFAULT_PAGE_SIZE = 50
AREA_CREATE_OPTION_VALUE = "__create_new_area__"
CONFIG_SYNC_CONCURRENCY = 8
CONFIG_KINDS = {"automation", "script", "scene"}
REFERENCE_KEYS = {"entity_id", "device_id", "area_id", "floor_id", "label_id", "scene", "script"}
READINESS_STATUSES = {"ready", "needs_review", "blocked"}
WORKFLOW_STATUSES = {"open", "applied_pending_recheck", "skipped", "error"}
REVIEW_STATUSES = {"pending", "accepted", "rejected"}
GENERATION_STATUSES = {"success", "error"}
LLM_PROVIDER_KINDS = {"openai_compatible"}
SUGGESTION_RUN_STATUSES = {"queued", "running", "succeeded", "failed"}
SUGGESTION_PROPOSAL_STATUSES = {"proposed", "accepted", "rejected", "invalid"}
SUGGESTION_MAX_TARGETS_PER_RUN = 25
SUGGESTION_MAX_PATCH_OPS = 12
SUGGESTION_WORKER: SuggestionWorker | None = None


def app_name() -> str:
    return os.getenv("APP_NAME", "HA Entity Vault")


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


def log_event(event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {
        "event": event,
        "timestamp": utcnow().isoformat(),
        **fields,
    }
    logger.info(json.dumps(payload, default=str, separators=(",", ":")))


def normalize_next_url(next_url: str) -> str:
    if next_url.startswith("/"):
        return next_url
    return "/settings"


def get_current_url(request: Request) -> str:
    query = request.url.query
    if query:
        return f"{request.url.path}?{query}"
    return request.url.path


def set_active_profile_id(request: Request, profile_id: int | None) -> None:
    if profile_id is None:
        request.session.pop(ACTIVE_PROFILE_SESSION_KEY, None)
        return
    request.session[ACTIVE_PROFILE_SESSION_KEY] = profile_id


def get_active_profile_id(request: Request) -> int | None:
    raw = request.session.get(ACTIVE_PROFILE_SESSION_KEY)
    if isinstance(raw, int):
        return raw
    return None


def with_profile_id(next_url: str, profile_id: int | None) -> str:
    normalized = normalize_next_url(next_url)
    split_result = urlsplit(normalized)
    query_items = [
        (k, v)
        for k, v in parse_qsl(split_result.query, keep_blank_values=True)
        if k != "profile_id"
    ]
    if profile_id is not None:
        query_items.append(("profile_id", str(profile_id)))
    rebuilt_query = urlencode(query_items)
    rebuilt_path = split_result.path or "/"
    return urlunsplit(("", "", rebuilt_path, rebuilt_query, ""))


def set_flash(request: Request, level: str, message: str) -> None:
    request.session[FLASH_SESSION_KEY] = {"level": level, "message": message}


def pop_flash(request: Request) -> dict[str, str] | None:
    raw = request.session.pop(FLASH_SESSION_KEY, None)
    if not isinstance(raw, dict):
        return None
    level = raw.get("level")
    message = raw.get("message")
    if not isinstance(level, str) or not isinstance(message, str):
        return None
    return {"level": level, "message": message}


def get_csrf_token(request: Request) -> str:
    token = request.session.get(CSRF_SESSION_KEY)
    if isinstance(token, str) and token:
        return token

    token = secrets.token_urlsafe(32)
    request.session[CSRF_SESSION_KEY] = token
    return token


def verify_csrf(request: Request, token: str) -> None:
    expected = request.session.get(CSRF_SESSION_KEY)
    if not isinstance(expected, str) or token != expected:
        raise HTTPException(status_code=400, detail="Invalid CSRF token")


def parse_ha_datetime(raw_value: Any) -> datetime | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None

    normalized = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def safe_json_dump(raw_value: Any) -> str:
    try:
        return json.dumps(raw_value, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError):
        return json.dumps({"_error": "failed_to_serialize"})


def safe_pretty_json(raw_json: str | None) -> str:
    if not raw_json:
        return "{}"
    try:
        return json.dumps(json.loads(raw_json), indent=2, sort_keys=True)
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw_json


def safe_json_load(raw_json: str | None, default: Any) -> Any:
    if not raw_json:
        return default
    try:
        return json.loads(raw_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def as_clean_string(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    cleaned = str(raw_value).strip()
    return cleaned or None


def normalize_string_list(raw_value: Any) -> list[str]:
    if isinstance(raw_value, (list, tuple, set)):
        values = raw_value
    else:
        return []

    normalized: list[str] = []
    for item in values:
        cleaned = as_clean_string(item)
        if cleaned:
            normalized.append(cleaned)
    return normalized


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, list) and not value:
            continue
        compacted[key] = value
    return compacted


def build_registry_lookup(
    registry_metadata: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]] | dict[str, str]]:
    areas_by_id: dict[str, dict[str, Any]] = {}
    devices_by_id: dict[str, dict[str, Any]] = {}
    entities_by_entity_id: dict[str, dict[str, Any]] = {}
    floors_by_id: dict[str, dict[str, Any]] = {}
    label_names_by_id: dict[str, str] = {}

    for item in registry_metadata.get("areas", []):
        area_id = as_clean_string(item.get("area_id") or item.get("id"))
        if area_id:
            areas_by_id[area_id] = item

    for item in registry_metadata.get("devices", []):
        device_id = as_clean_string(item.get("id") or item.get("device_id"))
        if device_id:
            devices_by_id[device_id] = item

    for item in registry_metadata.get("entities", []):
        entity_id = as_clean_string(item.get("entity_id"))
        if entity_id:
            entities_by_entity_id[entity_id] = item

    for item in registry_metadata.get("floors", []):
        floor_id = as_clean_string(item.get("floor_id") or item.get("id"))
        if floor_id:
            floors_by_id[floor_id] = item

    for item in registry_metadata.get("labels", []):
        label_id = as_clean_string(item.get("label_id") or item.get("id"))
        label_name = as_clean_string(item.get("name"))
        if label_id and label_name:
            label_names_by_id[label_id] = label_name

    return {
        "areas_by_id": areas_by_id,
        "devices_by_id": devices_by_id,
        "entities_by_entity_id": entities_by_entity_id,
        "floors_by_id": floors_by_id,
        "label_names_by_id": label_names_by_id,
    }


def enrich_state_snapshot(
    state: dict[str, Any],
    lookup: dict[str, dict[str, dict[str, Any]] | dict[str, str]],
) -> dict[str, Any]:
    attributes_raw = state.get("attributes")
    attributes = attributes_raw if isinstance(attributes_raw, dict) else {}
    entity_id = as_clean_string(state.get("entity_id")) or ""

    entities_by_entity_id = lookup["entities_by_entity_id"]
    devices_by_id = lookup["devices_by_id"]
    areas_by_id = lookup["areas_by_id"]
    floors_by_id = lookup["floors_by_id"]
    label_names_by_id = lookup["label_names_by_id"]

    entity_entry = (
        entities_by_entity_id.get(entity_id, {})
        if isinstance(entities_by_entity_id, dict)
        else {}
    )
    if not isinstance(entity_entry, dict):
        entity_entry = {}

    device_id = as_clean_string(entity_entry.get("device_id")) or as_clean_string(
        attributes.get("device_id")
    )
    device_entry: dict[str, Any] = {}
    if isinstance(devices_by_id, dict) and device_id:
        raw_device = devices_by_id.get(device_id)
        if isinstance(raw_device, dict):
            device_entry = raw_device

    area_id = (
        as_clean_string(entity_entry.get("area_id"))
        or as_clean_string(attributes.get("area_id"))
        or as_clean_string(device_entry.get("area_id"))
    )
    area_entry: dict[str, Any] = {}
    if isinstance(areas_by_id, dict) and area_id:
        raw_area = areas_by_id.get(area_id)
        if isinstance(raw_area, dict):
            area_entry = raw_area

    floor_id = (
        as_clean_string(area_entry.get("floor_id"))
        or as_clean_string(attributes.get("floor_id"))
        or as_clean_string(device_entry.get("floor_id"))
    )
    floor_entry: dict[str, Any] = {}
    if isinstance(floors_by_id, dict) and floor_id:
        raw_floor = floors_by_id.get(floor_id)
        if isinstance(raw_floor, dict):
            floor_entry = raw_floor

    friendly_name = as_clean_string(attributes.get("friendly_name")) or as_clean_string(
        entity_entry.get("name")
    )
    if not friendly_name:
        friendly_name = as_clean_string(entity_entry.get("original_name"))

    device_name = as_clean_string(device_entry.get("name_by_user")) or as_clean_string(
        device_entry.get("name")
    )

    area_name = as_clean_string(area_entry.get("name")) or as_clean_string(
        attributes.get("area_name")
    )
    floor_name = as_clean_string(floor_entry.get("name")) or as_clean_string(
        attributes.get("floor_name")
    )

    location_name: str | None
    if area_name and floor_name:
        location_name = f"{area_name} ({floor_name})"
    else:
        location_name = area_name or floor_name

    entity_labels = normalize_string_list(entity_entry.get("labels"))
    device_labels = normalize_string_list(device_entry.get("labels"))
    attribute_labels = normalize_string_list(attributes.get("label_ids"))
    label_ids = unique_preserving_order(entity_labels + device_labels + attribute_labels)

    label_names: list[str] = []
    if isinstance(label_names_by_id, dict):
        label_names = unique_preserving_order(
            [label_names_by_id.get(label_id, label_id) for label_id in label_ids]
        )

    labels_payload: dict[str, Any] | None = None
    if label_ids or label_names:
        labels_payload = {"ids": label_ids, "names": label_names}

    metadata = compact_dict(
        {
            "entity_registry_id": as_clean_string(entity_entry.get("id")),
            "entity_registry_name": as_clean_string(entity_entry.get("name")),
            "entity_original_name": as_clean_string(entity_entry.get("original_name")),
            "entity_platform": as_clean_string(entity_entry.get("platform")),
            "entity_disabled_by": as_clean_string(entity_entry.get("disabled_by")),
            "entity_hidden_by": as_clean_string(entity_entry.get("hidden_by")),
            "entity_has_entity_name": entity_entry.get("has_entity_name"),
            "entity_unique_id": as_clean_string(entity_entry.get("unique_id")),
            "device_manufacturer": as_clean_string(device_entry.get("manufacturer")),
            "device_model": as_clean_string(device_entry.get("model")),
            "device_sw_version": as_clean_string(device_entry.get("sw_version")),
            "device_hw_version": as_clean_string(device_entry.get("hw_version")),
            "device_via_device_id": as_clean_string(device_entry.get("via_device_id")),
            "attribute_device_class": as_clean_string(attributes.get("device_class")),
            "attribute_state_class": as_clean_string(attributes.get("state_class")),
            "attribute_unit_of_measurement": as_clean_string(
                attributes.get("unit_of_measurement")
            ),
            "attribute_icon": as_clean_string(attributes.get("icon")),
            "attribute_entity_picture": as_clean_string(attributes.get("entity_picture")),
            "attribute_supported_features": attributes.get("supported_features"),
            "labels": labels_payload,
        }
    )

    return {
        "attributes": attributes,
        "friendly_name": friendly_name,
        "device_id": device_id,
        "device_name": device_name,
        "area_id": area_id,
        "area_name": area_name,
        "floor_id": floor_id,
        "floor_name": floor_name,
        "location_name": location_name,
        "labels_json": safe_json_dump(labels_payload) if labels_payload is not None else None,
        "metadata_json": safe_json_dump(metadata) if metadata else None,
    }


def resolve_profile_token(profile: Profile) -> str:
    if profile.token_env_var:
        value = os.getenv(profile.token_env_var)
        if value:
            return value

    if profile.token:
        return profile.token

    fallback = os.getenv("HA_TOKEN", "")
    return fallback


def get_enabled_profiles(session: Session) -> list[Profile]:
    return list(
        session.exec(
            select(Profile)
            .where(Profile.is_enabled.is_(True))
            .order_by(Profile.name, Profile.id)
        ).all()
    )


def get_first_enabled_profile(session: Session) -> Profile | None:
    return session.exec(
        select(Profile)
        .where(Profile.is_enabled.is_(True))
        .order_by(Profile.name, Profile.id)
    ).first()


def choose_active_profile(
    session: Session,
    request: Request,
    profile_id: int | None,
) -> Profile | None:
    if profile_id is not None:
        requested_profile = session.get(Profile, profile_id)
        if requested_profile is not None and requested_profile.is_enabled:
            set_active_profile_id(request, requested_profile.id)
            return requested_profile

    session_profile_id = get_active_profile_id(request)
    if session_profile_id is not None:
        session_profile = session.get(Profile, session_profile_id)
        if session_profile is not None and session_profile.is_enabled:
            return session_profile
        set_active_profile_id(request, None)

    first_enabled = get_first_enabled_profile(session)
    if first_enabled is not None:
        set_active_profile_id(request, first_enabled.id)
    return first_enabled


def require_enabled_profile(profile: Profile, request: Request) -> bool:
    if profile.is_enabled:
        return True
    set_flash(request, "error", f"Profile '{profile.name}' is disabled. Re-enable it from settings.")
    return False


def get_latest_sync_run(session: Session, profile_id: int) -> SyncRun | None:
    stmt = (
        select(SyncRun)
        .where(SyncRun.profile_id == profile_id)
        .where(SyncRun.status == "success")
        .order_by(SyncRun.pulled_at.desc(), SyncRun.id.desc())
    )
    return session.exec(stmt).first()


def build_query(**kwargs: Any) -> str:
    filtered: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        filtered[key] = value
    return urlencode(filtered)


def build_entity_stmt(
    profile_id: int,
    sync_run_id: int,
    q: str,
    domain: str,
    state_value: str,
    changed_within: int | None,
):
    stmt = select(EntitySnapshot).where(
        EntitySnapshot.profile_id == profile_id,
        EntitySnapshot.sync_run_id == sync_run_id,
    )

    if q:
        pattern = f"%{q.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(EntitySnapshot.entity_id).like(pattern),
                func.lower(func.coalesce(EntitySnapshot.friendly_name, "")).like(pattern),
                func.lower(func.coalesce(EntitySnapshot.device_name, "")).like(pattern),
                func.lower(func.coalesce(EntitySnapshot.area_name, "")).like(pattern),
                func.lower(func.coalesce(EntitySnapshot.location_name, "")).like(pattern),
                func.lower(func.coalesce(EntitySnapshot.labels_json, "")).like(pattern),
            )
        )

    if domain:
        stmt = stmt.where(EntitySnapshot.domain == domain)

    if state_value:
        stmt = stmt.where(EntitySnapshot.state == state_value)

    if changed_within is not None:
        cutoff = utcnow() - timedelta(minutes=changed_within)
        stmt = stmt.where(EntitySnapshot.last_updated.is_not(None))
        stmt = stmt.where(EntitySnapshot.last_updated >= cutoff)

    return stmt


def config_kind_from_entity_id(entity_id: str) -> str | None:
    if "." not in entity_id:
        return None
    kind = entity_id.split(".", 1)[0]
    if kind not in CONFIG_KINDS:
        return None
    return kind


def as_list(raw_value: Any) -> list[Any]:
    if isinstance(raw_value, list):
        return raw_value
    if raw_value is None:
        return []
    return [raw_value]


def config_value_count(
    config: dict[str, Any],
    primary_key: str,
    legacy_key: str | None = None,
) -> int:
    for key in [primary_key, legacy_key]:
        if key is None:
            continue
        if key not in config:
            continue
        return len(as_list(config.get(key)))
    return 0


def derive_config_key(
    kind: str,
    entity_id: str,
    state: dict[str, Any] | None,
    registry_entry: dict[str, Any] | None,
) -> str | None:
    attributes: dict[str, Any] = {}
    if isinstance(state, dict):
        raw_attributes = state.get("attributes")
        if isinstance(raw_attributes, dict):
            attributes = raw_attributes

    if kind in {"automation", "scene"}:
        key = as_clean_string(attributes.get("id"))
        if key:
            return key

    if kind == "script" and "." in entity_id:
        object_id = as_clean_string(entity_id.split(".", 1)[1])
        if object_id:
            return object_id

    if isinstance(registry_entry, dict):
        return as_clean_string(registry_entry.get("unique_id"))
    return None


def build_config_candidates(
    states: list[dict[str, Any]],
    registry_metadata: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    states_by_entity_id: dict[str, dict[str, Any]] = {}
    for state_item in states:
        entity_id = as_clean_string(state_item.get("entity_id"))
        if not entity_id:
            continue
        kind = config_kind_from_entity_id(entity_id)
        if kind is None:
            continue
        states_by_entity_id[entity_id] = state_item

    registry_entities_by_id: dict[str, dict[str, Any]] = {}
    for entry in registry_metadata.get("entities", []):
        entity_id = as_clean_string(entry.get("entity_id"))
        if not entity_id:
            continue
        kind = config_kind_from_entity_id(entity_id)
        if kind is None:
            continue
        registry_entities_by_id[entity_id] = entry

    candidates: list[dict[str, Any]] = []
    for entity_id in sorted(set(states_by_entity_id) | set(registry_entities_by_id)):
        kind = config_kind_from_entity_id(entity_id)
        if kind is None:
            continue

        state_entry = states_by_entity_id.get(entity_id)
        registry_entry = registry_entities_by_id.get(entity_id)
        state_name: str | None = None
        if isinstance(state_entry, dict):
            raw_attributes = state_entry.get("attributes")
            if isinstance(raw_attributes, dict):
                state_name = as_clean_string(raw_attributes.get("friendly_name"))

        registry_name: str | None = None
        if isinstance(registry_entry, dict):
            registry_name = as_clean_string(registry_entry.get("name")) or as_clean_string(
                registry_entry.get("original_name")
            )

        candidates.append(
            {
                "kind": kind,
                "entity_id": entity_id,
                "state": state_entry,
                "registry_entry": registry_entry,
                "name": state_name or registry_name,
                "config_key": derive_config_key(kind, entity_id, state_entry, registry_entry),
            }
        )

    return candidates


def extract_reference_values(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        cleaned = as_clean_string(raw_value)
        return [cleaned] if cleaned else []

    if isinstance(raw_value, (list, tuple, set)):
        values: list[str] = []
        for item in raw_value:
            values.extend(extract_reference_values(item))
        return values

    return []


def extract_config_references(raw_value: Any) -> dict[str, list[str]]:
    collected: dict[str, list[str]] = {key: [] for key in REFERENCE_KEYS}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in REFERENCE_KEYS:
                    collected[key].extend(extract_reference_values(value))
                visit(value)
            return
        if isinstance(node, list):
            for item in node:
                visit(item)

    visit(raw_value)

    references: dict[str, list[str]] = {}
    for key, values in collected.items():
        deduped = unique_preserving_order(values)
        if deduped:
            references[key] = deduped
    return references


def summarize_config(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    use_blueprint = config.get("use_blueprint")
    blueprint_path: str | None = None
    if isinstance(use_blueprint, dict):
        blueprint_path = as_clean_string(use_blueprint.get("path"))

    summary = compact_dict(
        {
            "name": as_clean_string(config.get("alias")) or as_clean_string(config.get("name")),
            "description": as_clean_string(config.get("description")),
            "mode": as_clean_string(config.get("mode")),
            "uses_blueprint": isinstance(use_blueprint, dict),
            "blueprint_path": blueprint_path,
        }
    )

    if kind == "automation":
        summary["trigger_count"] = config_value_count(config, "triggers", "trigger")
        summary["condition_count"] = config_value_count(config, "conditions", "condition")
        summary["action_count"] = config_value_count(config, "actions", "action")
    elif kind == "script":
        summary["sequence_count"] = config_value_count(config, "sequence")
    elif kind == "scene":
        entities = config.get("entities")
        if isinstance(entities, dict):
            summary["entity_count"] = len(entities)
        else:
            summary["entity_count"] = len(as_list(entities))

    return summary


def get_latest_config_sync_run(session: Session, profile_id: int) -> ConfigSyncRun | None:
    stmt = (
        select(ConfigSyncRun)
        .where(ConfigSyncRun.profile_id == profile_id)
        .where(ConfigSyncRun.status.in_(["success", "partial"]))
        .order_by(ConfigSyncRun.pulled_at.desc(), ConfigSyncRun.id.desc())
    )
    return session.exec(stmt).first()


def get_latest_suggestion_run(session: Session, profile_id: int) -> EntitySuggestionRun | None:
    stmt = (
        select(EntitySuggestionRun)
        .where(EntitySuggestionRun.profile_id == profile_id)
        .order_by(EntitySuggestionRun.pulled_at.desc(), EntitySuggestionRun.id.desc())
    )
    return session.exec(stmt).first()


def get_latest_draft_run(session: Session, profile_id: int) -> AutomationDraftRun | None:
    stmt = (
        select(AutomationDraftRun)
        .where(AutomationDraftRun.profile_id == profile_id)
        .order_by(AutomationDraftRun.pulled_at.desc(), AutomationDraftRun.id.desc())
    )
    return session.exec(stmt).first()


def get_profile_llm_connections(
    session: Session,
    profile_id: int,
    enabled_only: bool = False,
) -> list[LLMConnection]:
    stmt = select(LLMConnection).where(LLMConnection.profile_id == profile_id)
    if enabled_only:
        stmt = stmt.where(LLMConnection.is_enabled.is_(True))
    stmt = stmt.order_by(LLMConnection.is_default.desc(), LLMConnection.name, LLMConnection.id)
    return list(session.exec(stmt).all())


def get_default_llm_connection(session: Session, profile_id: int) -> LLMConnection | None:
    defaults = session.exec(
        select(LLMConnection)
        .where(LLMConnection.profile_id == profile_id, LLMConnection.is_enabled.is_(True))
        .order_by(LLMConnection.is_default.desc(), LLMConnection.name, LLMConnection.id)
    ).all()
    return defaults[0] if defaults else None


def parse_extra_headers_json(raw_json: str) -> dict[str, str]:
    cleaned = raw_json.strip()
    if not cleaned:
        return {}
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Extra headers must be valid JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Extra headers must be a JSON object.")
    headers: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        normalized_key = key.strip()
        if not normalized_key:
            continue
        headers[normalized_key] = str(value).strip()
    return headers


def resolve_llm_api_key(connection: LLMConnection) -> str:
    if connection.api_key_env_var:
        value = os.getenv(connection.api_key_env_var)
        if value:
            return value.strip()
    return ""


def build_llm_settings_from_connection(
    connection: LLMConnection,
    api_key: str,
) -> LLMSettings:
    try:
        extra_headers = parse_extra_headers_json(connection.extra_headers_json or "")
    except ValueError:
        extra_headers = {}
    return LLMSettings(
        enabled=True,
        base_url=connection.base_url,
        api_key=api_key,
        model=connection.model,
        timeout_seconds=connection.timeout_seconds,
        max_concurrency=1,
        allow_missing_api_key=True,
        temperature=connection.temperature,
        max_output_tokens=connection.max_output_tokens,
        extra_headers=extra_headers if extra_headers else None,
    )


def sanitize_text(raw_text: str, secrets: list[str]) -> str:
    sanitized = raw_text
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, "***redacted***")
    return sanitized


def sanitize_payload(payload: Any, secrets: list[str]) -> Any:
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            sanitized[key] = sanitize_payload(value, secrets)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_payload(value, secrets) for value in payload]
    if isinstance(payload, str):
        return sanitize_text(payload, secrets)
    return payload


def build_suggestion_targets_for_run(
    session: Session,
    run: SuggestionRun,
) -> list[ConfigSnapshot]:
    filters = safe_json_load(run.filters_json, {})
    if not isinstance(filters, dict):
        filters = {}

    snapshot_ids_raw = filters.get("snapshot_ids")
    if isinstance(snapshot_ids_raw, list):
        snapshot_ids: list[int] = []
        for item in snapshot_ids_raw:
            if isinstance(item, int):
                snapshot_ids.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                snapshot_ids.append(int(item.strip()))
        if snapshot_ids:
            rows = session.exec(
                select(ConfigSnapshot).where(
                    ConfigSnapshot.profile_id == run.profile_id,
                    ConfigSnapshot.id.in_(snapshot_ids),
                )
            ).all()
            return [
                row
                for row in rows
                if row.kind == "automation" and row.fetch_status == "success"
            ][:SUGGESTION_MAX_TARGETS_PER_RUN]

    config_sync_run_id = run.config_sync_run_id
    if config_sync_run_id is None:
        latest = get_latest_config_sync_run(session, run.profile_id)
        if latest is None:
            return []
        config_sync_run_id = latest.id
    if config_sync_run_id is None:
        return []

    q = str(filters.get("q", "")).strip()
    stmt = (
        select(ConfigSnapshot)
        .where(
            ConfigSnapshot.profile_id == run.profile_id,
            ConfigSnapshot.config_sync_run_id == config_sync_run_id,
            ConfigSnapshot.kind == "automation",
            ConfigSnapshot.fetch_status == "success",
        )
        .order_by(ConfigSnapshot.entity_id)
    )
    if q:
        pattern = f"%{q.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(ConfigSnapshot.entity_id).like(pattern),
                func.lower(func.coalesce(ConfigSnapshot.name, "")).like(pattern),
                func.lower(func.coalesce(ConfigSnapshot.summary_json, "")).like(pattern),
            )
        )

    max_targets_raw = filters.get("max_targets")
    max_targets = SUGGESTION_MAX_TARGETS_PER_RUN
    if isinstance(max_targets_raw, int):
        max_targets = max(1, min(SUGGESTION_MAX_TARGETS_PER_RUN, max_targets_raw))
    return list(session.exec(stmt.limit(max_targets)).all())


def create_suggestion_audit_event(
    session: Session,
    profile_id: int,
    event_type: str,
    actor: str,
    suggestion_run_id: int | None = None,
    proposal_id: int | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    event = SuggestionAuditEvent(
        profile_id=profile_id,
        suggestion_run_id=suggestion_run_id,
        proposal_id=proposal_id,
        event_type=event_type,
        actor=actor,
        payload_json=safe_json_dump(payload) if payload else None,
        created_at=utcnow(),
    )
    session.add(event)


def build_config_item_stmt(
    profile_id: int,
    config_sync_run_id: int,
    q: str,
    kind: str,
    status: str,
):
    stmt = select(ConfigSnapshot).where(
        ConfigSnapshot.profile_id == profile_id,
        ConfigSnapshot.config_sync_run_id == config_sync_run_id,
    )

    if q:
        pattern = f"%{q.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(ConfigSnapshot.entity_id).like(pattern),
                func.lower(func.coalesce(ConfigSnapshot.name, "")).like(pattern),
                func.lower(func.coalesce(ConfigSnapshot.config_key, "")).like(pattern),
                func.lower(func.coalesce(ConfigSnapshot.fetch_error, "")).like(pattern),
                func.lower(func.coalesce(ConfigSnapshot.summary_json, "")).like(pattern),
                func.lower(func.coalesce(ConfigSnapshot.references_json, "")).like(pattern),
            )
        )

    if kind:
        stmt = stmt.where(ConfigSnapshot.kind == kind)

    if status:
        stmt = stmt.where(ConfigSnapshot.fetch_status == status)

    return stmt


def build_entity_suggestion_stmt(
    profile_id: int,
    suggestion_run_id: int,
    q: str,
    readiness_status: str,
    domain: str,
):
    stmt = select(EntitySuggestion).where(
        EntitySuggestion.profile_id == profile_id,
        EntitySuggestion.suggestion_run_id == suggestion_run_id,
    )

    if q:
        pattern = f"%{q.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(EntitySuggestion.entity_id).like(pattern),
                func.lower(func.coalesce(EntitySuggestion.issues_json, "")).like(pattern),
                func.lower(func.coalesce(EntitySuggestion.missing_fields_json, "")).like(pattern),
                func.lower(func.coalesce(EntitySuggestion.llm_suggestions_json, "")).like(pattern),
            )
        )

    if readiness_status:
        stmt = stmt.where(EntitySuggestion.readiness_status == readiness_status)

    if domain:
        stmt = stmt.where(EntitySuggestion.domain == domain)

    return stmt


def parse_issue_list(raw_json: str | None) -> list[dict[str, Any]]:
    parsed = safe_json_load(raw_json, [])
    if not isinstance(parsed, list):
        return []
    issues: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            issues.append(item)
    return issues


def split_suggestion_issues(
    suggestion: EntitySuggestion,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues = parse_issue_list(suggestion.issues_json)
    return split_issues_by_fixability(issues)


def suggestion_has_fixable_issues(suggestion: EntitySuggestion) -> bool:
    fixable, _ = split_suggestion_issues(suggestion)
    return bool(fixable)


def issue_codes_for_suggestion(suggestion: EntitySuggestion) -> set[str]:
    codes: set[str] = set()
    for issue in parse_issue_list(suggestion.issues_json):
        code_raw = issue.get("code")
        if isinstance(code_raw, str):
            code = code_raw.strip()
            if code:
                codes.add(code)
    return codes


def parse_label_ids_from_snapshot(snapshot: EntitySnapshot | None) -> list[str]:
    if snapshot is None:
        return []
    payload = safe_json_load(snapshot.labels_json, {})
    if not isinstance(payload, dict):
        return []
    raw_ids = payload.get("ids")
    if not isinstance(raw_ids, list):
        return []
    label_ids: list[str] = []
    for item in raw_ids:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                label_ids.append(cleaned)
    return unique_preserving_order(label_ids)


def parse_label_names_from_snapshot(snapshot: EntitySnapshot | None) -> list[str]:
    if snapshot is None:
        return []
    payload = safe_json_load(snapshot.labels_json, {})
    if not isinstance(payload, dict):
        return []
    raw_names = payload.get("names")
    if not isinstance(raw_names, list):
        return []
    label_names: list[str] = []
    for item in raw_names:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                label_names.append(cleaned)
    return unique_preserving_order(label_names)


def issue_code(issue: dict[str, Any]) -> str:
    raw_code = issue.get("code")
    if not isinstance(raw_code, str):
        return ""
    return raw_code.strip()


def build_workflow_issue_sections(
    suggestion: EntitySuggestion,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixable, manual_only = split_suggestion_issues(suggestion)
    fixable_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []

    for issue in fixable:
        row = dict(issue)
        row["code"] = issue_code(issue)
        fixable_rows.append(row)

    for issue in manual_only:
        row = dict(issue)
        row["code"] = issue_code(issue)
        row["guidance"] = manual_guidance_for_issue(issue)
        manual_rows.append(row)
    return fixable_rows, manual_rows


def normalize_label_ids_form(raw_values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in raw_values:
        value = raw.strip()
        if value:
            cleaned.append(value)
    return unique_preserving_order(cleaned)


def workflow_status_rank(status: str) -> int:
    if status == "blocked":
        return 0
    if status == "needs_review":
        return 1
    return 2


def build_automation_draft_stmt(
    profile_id: int,
    draft_run_id: int,
    q: str,
    review_status: str,
    template_id: str,
    generation_status: str,
):
    stmt = select(AutomationDraft).where(
        AutomationDraft.profile_id == profile_id,
        AutomationDraft.draft_run_id == draft_run_id,
    )

    if q:
        pattern = f"%{q.lower()}%"
        stmt = stmt.where(
            or_(
                func.lower(AutomationDraft.entity_id).like(pattern),
                func.lower(AutomationDraft.title).like(pattern),
                func.lower(func.coalesce(AutomationDraft.generation_error, "")).like(pattern),
                func.lower(func.coalesce(AutomationDraft.template_id, "")).like(pattern),
            )
        )

    if review_status:
        stmt = stmt.where(AutomationDraft.review_status == review_status)

    if template_id:
        stmt = stmt.where(AutomationDraft.template_id == template_id)

    if generation_status:
        stmt = stmt.where(AutomationDraft.generation_status == generation_status)

    return stmt


async def fetch_config_details(
    client: HAClient,
    kind: str,
    entity_id: str,
    config_key: str | None,
    has_state: bool,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if kind == "automation":
        if has_state:
            try:
                return await client.fetch_automation_config_ws(entity_id), None, "ws"
            except HAClientError as ws_exc:
                if config_key:
                    try:
                        return await client.fetch_automation_config(config_key), None, "rest"
                    except HAClientError as rest_exc:
                        return None, str(rest_exc), None
                return None, str(ws_exc), None
        if config_key:
            try:
                return await client.fetch_automation_config(config_key), None, "rest"
            except HAClientError as rest_exc:
                return None, str(rest_exc), None
        return None, "missing_config_locator", None

    if kind == "script":
        if has_state:
            try:
                return await client.fetch_script_config_ws(entity_id), None, "ws"
            except HAClientError as ws_exc:
                if config_key:
                    try:
                        return await client.fetch_script_config(config_key), None, "rest"
                    except HAClientError as rest_exc:
                        return None, str(rest_exc), None
                return None, str(ws_exc), None
        if config_key:
            try:
                return await client.fetch_script_config(config_key), None, "rest"
            except HAClientError as rest_exc:
                return None, str(rest_exc), None
        return None, "missing_config_locator", None

    if kind == "scene":
        if config_key:
            try:
                return await client.fetch_scene_config(config_key), None, "rest"
            except HAClientError as rest_exc:
                return None, str(rest_exc), None
        return None, "missing_config_locator", None

    return None, f"unsupported_kind:{kind}", None


async def build_config_snapshot_from_candidate(
    client: HAClient,
    semaphore: asyncio.Semaphore,
    profile_id: int,
    config_sync_run_id: int,
    pulled_at: datetime,
    candidate: dict[str, Any],
) -> ConfigSnapshot:
    async with semaphore:
        kind = str(candidate.get("kind", "")).strip()
        entity_id = str(candidate.get("entity_id", "")).strip()
        state = candidate.get("state")
        state_dict = state if isinstance(state, dict) else None
        registry_entry_raw = candidate.get("registry_entry")
        registry_entry = registry_entry_raw if isinstance(registry_entry_raw, dict) else None
        has_state = state_dict is not None
        config_key = as_clean_string(candidate.get("config_key"))

        config_payload, fetch_error, detail_source = await fetch_config_details(
            client=client,
            kind=kind,
            entity_id=entity_id,
            config_key=config_key,
            has_state=has_state,
        )

        name = as_clean_string(candidate.get("name"))
        if not name and isinstance(config_payload, dict):
            name = as_clean_string(config_payload.get("alias")) or as_clean_string(
                config_payload.get("name")
            )

        attributes: dict[str, Any] = {}
        state_value: str | None = None
        last_changed: datetime | None = None
        last_updated: datetime | None = None
        if state_dict is not None:
            raw_attributes = state_dict.get("attributes")
            if isinstance(raw_attributes, dict):
                attributes = raw_attributes
            state_value = as_clean_string(state_dict.get("state"))
            last_changed = parse_ha_datetime(state_dict.get("last_changed"))
            last_updated = parse_ha_datetime(state_dict.get("last_updated"))

        references_payload: dict[str, list[str]] = {}
        summary_payload: dict[str, Any] = {}
        if isinstance(config_payload, dict):
            references_payload = extract_config_references(config_payload)
            summary_payload = summarize_config(kind, config_payload)
        elif name:
            summary_payload = {"name": name}

        metadata_payload = compact_dict(
            {
                "detail_source": detail_source,
                "state_present": has_state,
                "registry_present": registry_entry is not None,
                "registry_unique_id": as_clean_string(
                    registry_entry.get("unique_id") if registry_entry else None
                ),
                "registry_platform": as_clean_string(
                    registry_entry.get("platform") if registry_entry else None
                ),
                "registry_disabled_by": as_clean_string(
                    registry_entry.get("disabled_by") if registry_entry else None
                ),
                "registry_hidden_by": as_clean_string(
                    registry_entry.get("hidden_by") if registry_entry else None
                ),
                "registry_entry": registry_entry,
            }
        )

        fetch_status = "success" if fetch_error is None else "error"
        return ConfigSnapshot(
            profile_id=profile_id,
            config_sync_run_id=config_sync_run_id,
            kind=kind,
            entity_id=entity_id,
            config_key=config_key,
            name=name,
            state=state_value,
            fetch_status=fetch_status,
            fetch_error=fetch_error,
            summary_json=safe_json_dump(summary_payload) if summary_payload else None,
            references_json=safe_json_dump(references_payload) if references_payload else None,
            config_json=safe_json_dump(config_payload) if isinstance(config_payload, dict) else None,
            attributes_json=safe_json_dump(attributes) if attributes else None,
            metadata_json=safe_json_dump(metadata_payload) if metadata_payload else None,
            last_changed=last_changed,
            last_updated=last_updated,
            pulled_at=pulled_at,
        )


async def build_entity_suggestion_from_snapshot(
    snapshot: EntitySnapshot,
    profile_id: int,
    sync_run_id: int,
    suggestion_run_id: int,
    pulled_at: datetime,
    area_required: bool,
    llm_client: OpenAICompatibleLLMClient | None,
    llm_semaphore: asyncio.Semaphore | None,
) -> tuple[EntitySuggestion, str | None]:
    evaluation = evaluate_entity_snapshot(snapshot, area_required=area_required)
    llm_suggestions_payload: dict[str, Any] | None = None
    llm_error: str | None = None

    if llm_client is not None and evaluation.readiness_status != "ready":
        try:
            if llm_semaphore is not None:
                async with llm_semaphore:
                    llm_suggestions_payload = await llm_client.suggest_entity_metadata(
                        build_llm_suggestion_payload(snapshot, evaluation)
                    )
            else:
                llm_suggestions_payload = await llm_client.suggest_entity_metadata(
                    build_llm_suggestion_payload(snapshot, evaluation)
                )
        except LLMClientError as exc:
            llm_error = str(exc)

    source_metadata = dict(evaluation.source_metadata)
    source_metadata["area_check_mode"] = "strict" if area_required else "degraded_no_enrichment"
    source_metadata["area_required"] = area_required
    if llm_error is not None:
        source_metadata["llm_error"] = llm_error

    suggestion = EntitySuggestion(
        profile_id=profile_id,
        sync_run_id=sync_run_id,
        suggestion_run_id=suggestion_run_id,
        entity_snapshot_id=snapshot.id if snapshot.id is not None else 0,
        entity_id=snapshot.entity_id,
        domain=snapshot.domain,
        readiness_status=evaluation.readiness_status,
        missing_fields_json=safe_json_dump(evaluation.missing_fields),
        issues_json=safe_json_dump(evaluation.issues),
        semantic_type_json=safe_json_dump(evaluation.semantic_type),
        llm_suggestions_json=(
            safe_json_dump(llm_suggestions_payload) if llm_suggestions_payload is not None else None
        ),
        source_metadata_json=safe_json_dump(source_metadata),
        pulled_at=pulled_at,
    )
    return suggestion, llm_error


async def perform_profile_sync(
    session: Session,
    profile: Profile,
    request_id: str | None = None,
) -> tuple[SyncRun | None, str | None]:
    token = resolve_profile_token(profile)
    if not token:
        return None, "No token configured in profile or environment."

    client = HAClient(
        base_url=profile.base_url,
        token=token,
        verify_tls=profile.verify_tls,
        timeout_seconds=profile.timeout_seconds,
    )

    pulled_at = utcnow()
    start = perf_counter()

    try:
        states = await client.fetch_states()
    except HAClientError as exc:
        failed_run = SyncRun(
            profile_id=profile.id if profile.id is not None else 0,
            pulled_at=pulled_at,
            entity_count=0,
            duration_ms=int((perf_counter() - start) * 1000),
            status="failed",
            error=str(exc),
        )
        session.add(failed_run)
        session.commit()
        log_event(
            "sync_failed",
            request_id=request_id,
            profile_id=profile.id,
            profile_name=profile.name,
            error=str(exc),
        )
        return None, f"Sync failed: {exc}"

    registry_metadata: dict[str, list[dict[str, Any]]] = {}
    try:
        registry_metadata = await client.fetch_registry_metadata()
    except HAClientError as exc:
        log_event(
            "sync_registry_enrichment_unavailable",
            request_id=request_id,
            profile_id=profile.id,
            profile_name=profile.name,
            error=str(exc),
        )

    registry_lookup = build_registry_lookup(registry_metadata)

    sync_run = SyncRun(
        profile_id=profile.id if profile.id is not None else 0,
        pulled_at=pulled_at,
        entity_count=len(states),
        duration_ms=0,
        status="success",
        error=None,
    )
    session.add(sync_run)
    session.commit()
    session.refresh(sync_run)

    sync_run_id = sync_run.id
    if sync_run_id is None:
        return None, "Sync run ID is unavailable."

    snapshots: list[EntitySnapshot] = []
    for state in states:
        entity_id = str(state.get("entity_id", "")).strip()
        if not entity_id:
            continue
        domain = entity_id.split(".", 1)[0] if "." in entity_id else "unknown"
        raw_context = state.get("context")
        enrichment = enrich_state_snapshot(state, registry_lookup)

        snapshots.append(
            EntitySnapshot(
                profile_id=profile.id if profile.id is not None else 0,
                sync_run_id=sync_run_id,
                entity_id=entity_id,
                domain=domain,
                state=str(state.get("state", "")),
                friendly_name=enrichment["friendly_name"],
                device_id=enrichment["device_id"],
                device_name=enrichment["device_name"],
                area_id=enrichment["area_id"],
                area_name=enrichment["area_name"],
                floor_id=enrichment["floor_id"],
                floor_name=enrichment["floor_name"],
                location_name=enrichment["location_name"],
                labels_json=enrichment["labels_json"],
                metadata_json=enrichment["metadata_json"],
                attributes_json=safe_json_dump(enrichment["attributes"]),
                context_json=safe_json_dump(raw_context) if raw_context is not None else None,
                last_changed=parse_ha_datetime(state.get("last_changed")),
                last_updated=parse_ha_datetime(state.get("last_updated")),
                pulled_at=pulled_at,
            )
        )

    session.add_all(snapshots)
    sync_run.duration_ms = int((perf_counter() - start) * 1000)
    sync_run.entity_count = len(snapshots)
    session.add(sync_run)
    session.commit()

    log_event(
        "sync_completed",
        request_id=request_id,
        profile_id=profile.id,
        profile_name=profile.name,
        sync_run_id=sync_run_id,
        entity_count=len(snapshots),
        duration_ms=sync_run.duration_ms,
        pulled_at=pulled_at.isoformat(),
    )
    return sync_run, None


async def perform_entity_suggestion_run(
    session: Session,
    profile: Profile,
    sync_run: SyncRun,
    request_id: str | None = None,
) -> tuple[EntitySuggestionRun, bool, int]:
    pulled_at = utcnow()
    start = perf_counter()
    llm_settings = load_llm_settings()
    use_llm = llm_is_configured(llm_settings)
    llm_client = OpenAICompatibleLLMClient(llm_settings) if use_llm else None
    llm_semaphore = asyncio.Semaphore(llm_settings.max_concurrency) if use_llm else None

    run = EntitySuggestionRun(
        profile_id=profile.id if profile.id is not None else 0,
        sync_run_id=sync_run.id if sync_run.id is not None else 0,
        pulled_at=pulled_at,
        entity_count=0,
        ready_count=0,
        needs_review_count=0,
        blocked_count=0,
        duration_ms=0,
        status="success",
        error=None,
        policy_version=SUGGESTION_POLICY_VERSION,
        llm_enabled=use_llm,
        llm_model=llm_settings.model if use_llm else None,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    run_id = run.id
    if run_id is None:
        raise HTTPException(status_code=500, detail="Suggestion run ID is unavailable")

    snapshots = list(
        session.exec(
            select(EntitySnapshot)
            .where(
                EntitySnapshot.profile_id == profile.id,
                EntitySnapshot.sync_run_id == sync_run.id,
            )
            .order_by(EntitySnapshot.entity_id)
        ).all()
    )
    candidates = [snapshot for snapshot in snapshots if is_supported_entity(snapshot)]
    has_any_area = any((item.area_id or item.area_name) for item in candidates)
    has_any_device = any((item.device_id or item.device_name) for item in candidates)
    area_required = (has_any_area or has_any_device) if candidates else True
    area_check_mode = "strict" if area_required else "degraded_no_enrichment"

    log_event(
        "entity_suggestion_run_started",
        request_id=request_id,
        profile_id=profile.id,
        profile_name=profile.name,
        sync_run_id=sync_run.id,
        suggestion_run_id=run_id,
        entity_count=len(candidates),
        llm_enabled=use_llm,
        area_check_mode=area_check_mode,
        has_any_area=has_any_area,
        has_any_device=has_any_device,
    )

    suggestions: list[EntitySuggestion] = []
    llm_errors: list[str] = []
    if candidates:
        results = await asyncio.gather(
            *[
                build_entity_suggestion_from_snapshot(
                    snapshot=snapshot,
                    profile_id=profile.id if profile.id is not None else 0,
                    sync_run_id=sync_run.id if sync_run.id is not None else 0,
                    suggestion_run_id=run_id,
                    pulled_at=pulled_at,
                    area_required=area_required,
                    llm_client=llm_client,
                    llm_semaphore=llm_semaphore,
                )
                for snapshot in candidates
            ]
        )
        suggestions = [item[0] for item in results]
        llm_errors = [item[1] for item in results if item[1]]

    run.entity_count = len(suggestions)
    run.ready_count = sum(1 for item in suggestions if item.readiness_status == "ready")
    run.needs_review_count = sum(1 for item in suggestions if item.readiness_status == "needs_review")
    run.blocked_count = sum(1 for item in suggestions if item.readiness_status == "blocked")
    run.duration_ms = int((perf_counter() - start) * 1000)

    if use_llm and llm_errors:
        run.status = "partial"
        run.error = f"llm_failures:{len(llm_errors)}"
    else:
        run.status = "success"
        run.error = None

    session.add_all(suggestions)
    session.add(run)
    session.commit()

    if run.status == "success":
        log_event(
            "entity_suggestion_run_completed",
            request_id=request_id,
            profile_id=profile.id,
            profile_name=profile.name,
            sync_run_id=sync_run.id,
            suggestion_run_id=run_id,
            entity_count=run.entity_count,
            ready_count=run.ready_count,
            needs_review_count=run.needs_review_count,
            blocked_count=run.blocked_count,
            duration_ms=run.duration_ms,
            area_check_mode=area_check_mode,
        )
    else:
        log_event(
            "entity_suggestion_run_failed",
            request_id=request_id,
            profile_id=profile.id,
            profile_name=profile.name,
            sync_run_id=sync_run.id,
            suggestion_run_id=run_id,
            error=run.error,
            llm_failures=len(llm_errors),
            area_check_mode=area_check_mode,
        )
    return run, area_required, len(llm_errors)


async def build_automation_draft_from_suggestion(
    suggestion: EntitySuggestion,
    entity_snapshot: EntitySnapshot,
    peer_snapshots: list[EntitySnapshot],
    profile_id: int,
    draft_run_id: int,
    suggestion_run_id: int,
    pulled_at: datetime,
    llm_client: OpenAICompatibleLLMClient | None,
    llm_semaphore: asyncio.Semaphore | None,
) -> AutomationDraft:
    semantic_type = safe_json_load(suggestion.semantic_type_json, {})
    if not isinstance(semantic_type, dict):
        semantic_type = {}

    template_id = pick_template_id(entity_snapshot.domain, semantic_type)
    if not template_id:
        return AutomationDraft(
            profile_id=profile_id,
            draft_run_id=draft_run_id,
            suggestion_run_id=suggestion_run_id,
            entity_suggestion_id=suggestion.id if suggestion.id is not None else 0,
            entity_id=entity_snapshot.entity_id,
            template_id="unknown",
            title=f"No template for {entity_snapshot.entity_id}",
            yaml_text=None,
            structured_json=None,
            rationale_json=safe_json_dump({"reason": "no_template_for_entity"}),
            generation_status="error",
            generation_error="no_template_for_entity",
            review_status="pending",
            pulled_at=pulled_at,
        )

    if llm_client is None:
        return AutomationDraft(
            profile_id=profile_id,
            draft_run_id=draft_run_id,
            suggestion_run_id=suggestion_run_id,
            entity_suggestion_id=suggestion.id if suggestion.id is not None else 0,
            entity_id=entity_snapshot.entity_id,
            template_id=template_id,
            title=f"{TEMPLATE_CATALOG[template_id]['title']} ({entity_snapshot.entity_id})",
            yaml_text=None,
            structured_json=None,
            rationale_json=safe_json_dump({"reason": "llm_disabled"}),
            generation_status="error",
            generation_error="llm_disabled",
            review_status="pending",
            pulled_at=pulled_at,
        )

    prompt_payload = build_draft_prompt_payload(
        entity_snapshot=entity_snapshot,
        semantic_type=semantic_type,
        template_id=template_id,
        peer_snapshots=peer_snapshots,
    )

    try:
        if llm_semaphore is not None:
            async with llm_semaphore:
                llm_response = await llm_client.generate_automation_draft(prompt_payload)
        else:
            llm_response = await llm_client.generate_automation_draft(prompt_payload)
        structured_payload = build_automation_structured_payload(
            llm_response=llm_response,
            template_id=template_id,
            entity_id=entity_snapshot.entity_id,
        )
        yaml_text = render_automation_yaml(structured_payload)
        return AutomationDraft(
            profile_id=profile_id,
            draft_run_id=draft_run_id,
            suggestion_run_id=suggestion_run_id,
            entity_suggestion_id=suggestion.id if suggestion.id is not None else 0,
            entity_id=entity_snapshot.entity_id,
            template_id=template_id,
            title=llm_response["title"],
            yaml_text=yaml_text,
            structured_json=safe_json_dump(structured_payload),
            rationale_json=safe_json_dump({"rationale": llm_response.get("rationale", "")}),
            generation_status="success",
            generation_error=None,
            review_status="pending",
            pulled_at=pulled_at,
        )
    except LLMClientError as exc:
        return AutomationDraft(
            profile_id=profile_id,
            draft_run_id=draft_run_id,
            suggestion_run_id=suggestion_run_id,
            entity_suggestion_id=suggestion.id if suggestion.id is not None else 0,
            entity_id=entity_snapshot.entity_id,
            template_id=template_id,
            title=f"{TEMPLATE_CATALOG[template_id]['title']} ({entity_snapshot.entity_id})",
            yaml_text=None,
            structured_json=None,
            rationale_json=safe_json_dump({"reason": "llm_error"}),
            generation_status="error",
            generation_error=str(exc),
            review_status="pending",
            pulled_at=pulled_at,
        )


async def process_suggestion_run_job(run_id: int) -> None:
    with Session(get_engine()) as session:
        run = session.get(SuggestionRun, run_id)
        if run is None:
            return
        if run.status not in {"queued", "running"}:
            return

        run.status = "running"
        run.started_at = run.started_at or utcnow()
        run.updated_at = utcnow()
        session.add(run)
        create_suggestion_audit_event(
            session,
            profile_id=run.profile_id,
            suggestion_run_id=run.id,
            event_type="suggestion_run_started",
            actor="system",
        )
        session.commit()

        profile = session.get(Profile, run.profile_id)
        connection = session.get(LLMConnection, run.llm_connection_id)
        if profile is None or connection is None:
            run.status = "failed"
            run.error = "profile_or_connection_missing"
            run.finished_at = utcnow()
            run.updated_at = utcnow()
            session.add(run)
            create_suggestion_audit_event(
                session,
                profile_id=run.profile_id,
                suggestion_run_id=run.id,
                event_type="suggestion_run_failed",
                actor="system",
                payload={"reason": run.error},
            )
            session.commit()
            return

        if not connection.is_enabled:
            run.status = "failed"
            run.error = "llm_connection_disabled"
            run.finished_at = utcnow()
            run.updated_at = utcnow()
            session.add(run)
            create_suggestion_audit_event(
                session,
                profile_id=run.profile_id,
                suggestion_run_id=run.id,
                event_type="suggestion_run_failed",
                actor="system",
                payload={"reason": run.error},
            )
            session.commit()
            return

        api_key = resolve_llm_api_key(connection)
        llm_settings = build_llm_settings_from_connection(connection, api_key)
        llm_client = OpenAICompatibleLLMClient(llm_settings)

        targets = build_suggestion_targets_for_run(session, run)
        run.target_count = len(targets)
        context_hash_source = safe_json_dump(
            {
                "filters": safe_json_load(run.filters_json, {}),
                "targets": [item.entity_id for item in targets],
            }
        )
        run.context_hash = hashlib.sha256(context_hash_source.encode("utf-8")).hexdigest()
        run.updated_at = utcnow()
        session.add(run)
        session.commit()

        if not targets:
            run.status = "failed"
            run.error = "no_eligible_automation_targets"
            run.finished_at = utcnow()
            run.updated_at = utcnow()
            session.add(run)
            create_suggestion_audit_event(
                session,
                profile_id=run.profile_id,
                suggestion_run_id=run.id,
                event_type="suggestion_run_failed",
                actor="system",
                payload={"reason": run.error},
            )
            session.commit()
            return

        known_entity_ids = build_known_entity_ids(session, run.profile_id, run.config_sync_run_id)
        for target in targets:
            known_entity_ids.add(target.entity_id.lower())

        secrets_to_redact = [api_key] if api_key else []
        system_prompt = (
            "You are a Home Assistant automation assistant. "
            "Return JSON only with a top-level key 'suggestions' that is an array. "
            "Each suggestion must include target_entity_id, summary, confidence, risk_level, "
            "proposed_patch (array), and verification_steps (array)."
        )

        for target in targets:
            context_payload = build_automation_context(session, profile, target)
            request_payload = {
                "schema_version": "haev.automation.suggestion.request.v1",
                "policy": {
                    "suggest_only": True,
                    "max_patch_ops": SUGGESTION_MAX_PATCH_OPS,
                },
                "context": context_payload,
            }

            try:
                llm_response = await llm_client.chat_json(system_prompt, request_payload)
                valid_items, validation_errors = validate_suggestion_payload(
                    llm_response,
                    known_entity_ids=known_entity_ids,
                    max_patch_ops=SUGGESTION_MAX_PATCH_OPS,
                )
                sanitized_response = sanitize_payload(llm_response, secrets_to_redact)
            except LLMClientError as exc:
                run.error_count += 1
                run.processed_count += 1
                run.updated_at = utcnow()
                session.add(
                    SuggestionProposal(
                        profile_id=run.profile_id,
                        suggestion_run_id=run.id if run.id is not None else 0,
                        config_snapshot_id=target.id,
                        target_entity_id=target.entity_id,
                        status="invalid",
                        schema_version="haev.automation.suggestion.v1",
                        summary=None,
                        confidence=None,
                        risk_level=None,
                        proposed_patch_json=None,
                        verification_steps_json=None,
                        raw_response_json=None,
                        validation_error=f"provider_error:{exc}",
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                )
                session.add(run)
                session.commit()
                continue

            if valid_items:
                for item in valid_items:
                    proposal = SuggestionProposal(
                        profile_id=run.profile_id,
                        suggestion_run_id=run.id if run.id is not None else 0,
                        config_snapshot_id=target.id,
                        target_entity_id=item["target_entity_id"],
                        status="proposed",
                        schema_version=item["schema_version"],
                        summary=item["summary"],
                        confidence=item["confidence"],
                        risk_level=item["risk_level"],
                        proposed_patch_json=safe_json_dump(item["proposed_patch"]),
                        verification_steps_json=safe_json_dump(item["verification_steps"]),
                        raw_response_json=safe_json_dump(sanitized_response),
                        validation_error=None,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                    session.add(proposal)
                    session.flush()
                    if proposal.id is not None:
                        create_suggestion_audit_event(
                            session,
                            profile_id=run.profile_id,
                            suggestion_run_id=run.id,
                            proposal_id=proposal.id,
                            event_type="suggestion_proposal_created",
                            actor="system",
                            payload={"target_entity_id": proposal.target_entity_id},
                        )
                    run.success_count += 1

            if validation_errors:
                run.invalid_count += 1
                invalid_proposal = SuggestionProposal(
                    profile_id=run.profile_id,
                    suggestion_run_id=run.id if run.id is not None else 0,
                    config_snapshot_id=target.id,
                    target_entity_id=target.entity_id,
                    status="invalid",
                    schema_version="haev.automation.suggestion.v1",
                    summary=None,
                    confidence=None,
                    risk_level=None,
                    proposed_patch_json=None,
                    verification_steps_json=None,
                    raw_response_json=safe_json_dump(sanitized_response),
                    validation_error=";".join(validation_errors),
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
                session.add(invalid_proposal)
                session.flush()
                if invalid_proposal.id is not None:
                    create_suggestion_audit_event(
                        session,
                        profile_id=run.profile_id,
                        suggestion_run_id=run.id,
                        proposal_id=invalid_proposal.id,
                        event_type="suggestion_proposal_invalid",
                        actor="system",
                        payload={"errors": validation_errors[:8]},
                    )

            run.processed_count += 1
            run.updated_at = utcnow()
            session.add(run)
            session.commit()

        run.finished_at = utcnow()
        run.updated_at = utcnow()
        if run.success_count > 0 or run.invalid_count > 0:
            run.status = "succeeded"
            run.error = None
        else:
            run.status = "failed"
            run.error = "No valid suggestions were generated."
        run.result_summary_json = safe_json_dump(
            {
                "target_count": run.target_count,
                "processed_count": run.processed_count,
                "success_count": run.success_count,
                "invalid_count": run.invalid_count,
                "error_count": run.error_count,
            }
        )
        session.add(run)
        create_suggestion_audit_event(
            session,
            profile_id=run.profile_id,
            suggestion_run_id=run.id,
            event_type="suggestion_run_completed" if run.status == "succeeded" else "suggestion_run_failed",
            actor="system",
            payload=safe_json_load(run.result_summary_json, {}),
        )
        session.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global SUGGESTION_WORKER

    configure_logging()
    run_migrations()

    worker = SuggestionWorker(process_suggestion_run_job)
    worker.start()
    SUGGESTION_WORKER = worker

    with Session(get_engine()) as session:
        pending = session.exec(
            select(SuggestionRun.id).where(SuggestionRun.status.in_(["queued", "running"]))
        ).all()
        run_ids = [int(item) for item in pending]
    if run_ids:
        worker.recover(run_ids)

    try:
        yield
    finally:
        await worker.stop()
        SUGGESTION_WORKER = None


def create_app() -> FastAPI:
    app = FastAPI(title=app_name(), lifespan=lifespan)

    app.add_middleware(
        SessionMiddleware,
        secret_key=os.getenv("SESSION_SECRET", "dev-only-change-me"),
        same_site="lax",
        https_only=False,
    )

    templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
    templates.env.globals["build_query"] = build_query

    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
        name="static",
    )

    def render_template(
        request: Request,
        template_name: str,
        context: dict[str, Any],
    ) -> HTMLResponse:
        context["request"] = request
        context["app_name"] = app_name()
        context["csrf_token"] = get_csrf_token(request)
        context["flash"] = pop_flash(request)
        context["now"] = utcnow()
        return templates.TemplateResponse(request, template_name, context)

    def with_navigation(
        request: Request,
        session: Session,
        context: dict[str, Any],
        active_profile: Profile | None,
    ) -> dict[str, Any]:
        context["nav_profiles"] = get_enabled_profiles(session)
        context["nav_active_profile"] = active_profile
        context["current_url"] = get_current_url(request)
        return context

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = int((perf_counter() - start) * 1000)
            log_event(
                "request_error",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=elapsed_ms,
                error=str(exc),
            )
            raise

        elapsed_ms = int((perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        log_event(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=elapsed_ms,
        )
        return response

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/", response_class=HTMLResponse)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/entities", status_code=303)

    @app.get("/settings", response_class=HTMLResponse)
    async def settings(
        request: Request,
        profile_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        profiles = list(
            session.exec(select(Profile).order_by(Profile.is_enabled.desc(), Profile.name, Profile.id)).all()
        )
        enabled_profiles = [profile for profile in profiles if profile.is_enabled]
        disabled_profiles = [profile for profile in profiles if not profile.is_enabled]
        llm_connections = list(
            session.exec(
                select(LLMConnection).order_by(
                    LLMConnection.profile_id,
                    LLMConnection.is_default.desc(),
                    LLMConnection.name,
                )
            ).all()
        )
        llm_connections_by_profile: dict[int, list[LLMConnection]] = {}
        for connection in llm_connections:
            llm_connections_by_profile.setdefault(connection.profile_id, []).append(connection)

        return render_template(
            request,
            "settings.html",
            with_navigation(
                request,
                session,
                {
                    "profiles": profiles,
                    "enabled_profiles": enabled_profiles,
                    "disabled_profiles": disabled_profiles,
                    "active_profile": active_profile,
                    "llm_connections_by_profile": llm_connections_by_profile,
                },
                active_profile,
            ),
        )

    @app.post("/profiles/select")
    async def select_profile(
        request: Request,
        profile_id: str = Form(default=""),
        next_url: str = Form(default="/entities"),
        csrf_token: str = Form(...),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        selected_profile: Profile | None = None
        normalized_profile_id = profile_id.strip()
        if normalized_profile_id:
            try:
                parsed_profile_id = int(normalized_profile_id)
            except ValueError:
                set_flash(request, "error", "Invalid profile selection.")
            else:
                candidate = session.get(Profile, parsed_profile_id)
                if candidate is not None and candidate.is_enabled:
                    selected_profile = candidate
                else:
                    set_flash(request, "error", "Selected profile is unavailable or disabled.")

        active_profile: Profile | None
        if selected_profile is not None:
            set_active_profile_id(request, selected_profile.id)
            active_profile = selected_profile
        else:
            active_profile = choose_active_profile(session, request, None)

        target_url = with_profile_id(next_url, active_profile.id if active_profile else None)
        return RedirectResponse(url=target_url, status_code=303)

    @app.post("/profiles")
    async def create_profile(
        request: Request,
        name: str = Form(...),
        base_url: str = Form(...),
        token: str = Form(default=""),
        token_env_var: str = Form(default="HA_TOKEN"),
        verify_tls: bool = Form(default=False),
        timeout_seconds: int = Form(default=10),
        csrf_token: str = Form(...),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        cleaned_name = name.strip()
        cleaned_url = base_url.strip().rstrip("/")
        cleaned_env_var = token_env_var.strip() or "HA_TOKEN"

        if not cleaned_name:
            set_flash(request, "error", "Profile name is required.")
            return RedirectResponse(url="/settings", status_code=303)
        if not cleaned_url.startswith("http://") and not cleaned_url.startswith("https://"):
            set_flash(request, "error", "Base URL must start with http:// or https://.")
            return RedirectResponse(url="/settings", status_code=303)

        existing = session.exec(select(Profile).where(Profile.name == cleaned_name)).first()
        if existing is not None:
            set_flash(request, "error", f"Profile '{cleaned_name}' already exists.")
            return RedirectResponse(url="/settings", status_code=303)

        profile = Profile(
            name=cleaned_name,
            base_url=cleaned_url,
            token=token.strip(),
            token_env_var=cleaned_env_var,
            verify_tls=verify_tls,
            timeout_seconds=max(1, min(timeout_seconds, 120)),
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(profile)
        session.commit()
        if get_active_profile_id(request) is None and profile.id is not None:
            set_active_profile_id(request, profile.id)

        set_flash(request, "success", f"Profile '{cleaned_name}' created.")
        return RedirectResponse(url="/settings", status_code=303)

    @app.post("/profiles/{profile_id}/update")
    async def update_profile(
        profile_id: int,
        request: Request,
        name: str = Form(...),
        base_url: str = Form(...),
        token: str = Form(default=""),
        token_env_var: str = Form(default="HA_TOKEN"),
        verify_tls: bool = Form(default=False),
        timeout_seconds: int = Form(default=10),
        clear_token: bool = Form(default=False),
        csrf_token: str = Form(...),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")

        cleaned_name = name.strip()
        cleaned_url = base_url.strip().rstrip("/")
        cleaned_env_var = token_env_var.strip() or "HA_TOKEN"

        duplicate = session.exec(
            select(Profile).where(Profile.name == cleaned_name, Profile.id != profile_id)
        ).first()
        if duplicate is not None:
            set_flash(request, "error", f"Profile name '{cleaned_name}' is already in use.")
            return RedirectResponse(url="/settings", status_code=303)

        if not cleaned_url.startswith("http://") and not cleaned_url.startswith("https://"):
            set_flash(request, "error", "Base URL must start with http:// or https://.")
            return RedirectResponse(url="/settings", status_code=303)

        profile.name = cleaned_name
        profile.base_url = cleaned_url
        profile.token_env_var = cleaned_env_var
        profile.verify_tls = verify_tls
        profile.timeout_seconds = max(1, min(timeout_seconds, 120))

        if clear_token:
            profile.token = ""
        elif token.strip():
            profile.token = token.strip()

        profile.updated_at = utcnow()

        session.add(profile)
        session.commit()

        set_flash(request, "success", f"Profile '{cleaned_name}' saved.")
        return RedirectResponse(url="/settings", status_code=303)

    @app.post("/profiles/{profile_id}/delete")
    async def delete_profile(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")

        deleted_name = profile.name
        was_active_profile = get_active_profile_id(request) == profile_id
        session.exec(delete(SuggestionAuditEvent).where(SuggestionAuditEvent.profile_id == profile_id))
        session.exec(delete(SuggestionProposal).where(SuggestionProposal.profile_id == profile_id))
        session.exec(delete(SuggestionRun).where(SuggestionRun.profile_id == profile_id))
        session.exec(delete(LLMConnection).where(LLMConnection.profile_id == profile_id))
        session.exec(delete(AutomationDraft).where(AutomationDraft.profile_id == profile_id))
        session.exec(delete(AutomationDraftRun).where(AutomationDraftRun.profile_id == profile_id))
        session.exec(delete(EntitySuggestion).where(EntitySuggestion.profile_id == profile_id))
        session.exec(delete(EntitySuggestionRun).where(EntitySuggestionRun.profile_id == profile_id))
        session.exec(delete(ConfigSnapshot).where(ConfigSnapshot.profile_id == profile_id))
        session.exec(delete(ConfigSyncRun).where(ConfigSyncRun.profile_id == profile_id))
        session.exec(delete(EntitySnapshot).where(EntitySnapshot.profile_id == profile_id))
        session.exec(delete(SyncRun).where(SyncRun.profile_id == profile_id))
        session.delete(profile)
        session.commit()

        if was_active_profile:
            choose_active_profile(session, request, None)

        set_flash(request, "success", f"Profile '{deleted_name}' deleted.")
        return RedirectResponse(url="/settings", status_code=303)

    @app.post("/profiles/{profile_id}/enable")
    async def enable_profile(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")

        if profile.is_enabled:
            set_flash(request, "success", f"Profile '{profile.name}' is already enabled.")
            return RedirectResponse(url="/settings", status_code=303)

        profile.is_enabled = True
        profile.updated_at = utcnow()
        session.add(profile)
        session.commit()

        if get_active_profile_id(request) is None:
            set_active_profile_id(request, profile.id)

        set_flash(request, "success", f"Profile '{profile.name}' enabled.")
        return RedirectResponse(url="/settings", status_code=303)

    @app.post("/profiles/{profile_id}/disable")
    async def disable_profile(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")

        if not profile.is_enabled:
            set_flash(request, "success", f"Profile '{profile.name}' is already disabled.")
            return RedirectResponse(url="/settings", status_code=303)

        profile.is_enabled = False
        profile.updated_at = utcnow()
        session.add(profile)
        session.commit()

        if get_active_profile_id(request) == profile_id:
            choose_active_profile(session, request, None)

        set_flash(request, "success", f"Profile '{profile.name}' disabled.")
        return RedirectResponse(url="/settings", status_code=303)

    @app.post("/profiles/{profile_id}/llm-connections")
    async def create_llm_connection(
        profile_id: int,
        request: Request,
        name: str = Form(...),
        base_url: str = Form(...),
        model: str = Form(...),
        api_key_env_var: str = Form(default=""),
        timeout_seconds: int = Form(default=20),
        temperature: float = Form(default=0.2),
        max_output_tokens: int = Form(default=900),
        extra_headers_json: str = Form(default=""),
        is_enabled: bool = Form(default=False),
        is_default: bool = Form(default=False),
        csrf_token: str = Form(...),
        next_url: str = Form(default="/settings"),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")

        cleaned_name = name.strip()
        cleaned_base_url = base_url.strip().rstrip("/")
        cleaned_model = model.strip()
        cleaned_env_var = api_key_env_var.strip()
        if not cleaned_name:
            set_flash(request, "error", "Connection name is required.")
            return RedirectResponse(url=with_profile_id(next_url, profile_id), status_code=303)
        if not cleaned_base_url.startswith("http://") and not cleaned_base_url.startswith("https://"):
            set_flash(request, "error", "LLM base URL must start with http:// or https://.")
            return RedirectResponse(url=with_profile_id(next_url, profile_id), status_code=303)
        if not cleaned_model:
            set_flash(request, "error", "LLM model is required.")
            return RedirectResponse(url=with_profile_id(next_url, profile_id), status_code=303)

        duplicate = session.exec(
            select(LLMConnection).where(
                LLMConnection.profile_id == profile_id,
                LLMConnection.name == cleaned_name,
            )
        ).first()
        if duplicate is not None:
            set_flash(request, "error", f"LLM connection '{cleaned_name}' already exists for this profile.")
            return RedirectResponse(url=with_profile_id(next_url, profile_id), status_code=303)

        try:
            parsed_headers = parse_extra_headers_json(extra_headers_json)
        except ValueError as exc:
            set_flash(request, "error", str(exc))
            return RedirectResponse(url=with_profile_id(next_url, profile_id), status_code=303)

        normalized_timeout = max(1, min(timeout_seconds, 300))
        normalized_temperature = max(0.0, min(float(temperature), 2.0))
        normalized_max_tokens = max(1, min(max_output_tokens, 8192))
        normalized_enabled = bool(is_enabled)
        normalized_default = bool(is_default and normalized_enabled)

        if normalized_default:
            existing_defaults = session.exec(
                select(LLMConnection).where(
                    LLMConnection.profile_id == profile_id,
                    LLMConnection.is_default.is_(True),
                )
            ).all()
            for item in existing_defaults:
                item.is_default = False
                item.updated_at = utcnow()
                session.add(item)

        connection = LLMConnection(
            profile_id=profile_id,
            name=cleaned_name,
            provider_kind="openai_compatible",
            base_url=cleaned_base_url,
            model=cleaned_model,
            api_key_env_var=cleaned_env_var or None,
            timeout_seconds=normalized_timeout,
            temperature=normalized_temperature,
            max_output_tokens=normalized_max_tokens,
            extra_headers_json=safe_json_dump(parsed_headers) if parsed_headers else None,
            is_enabled=normalized_enabled,
            is_default=normalized_default,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(connection)
        session.commit()

        set_flash(request, "success", f"LLM connection '{cleaned_name}' created.")
        return RedirectResponse(url=with_profile_id(next_url, profile_id), status_code=303)

    @app.post("/llm-connections/{connection_id}/update")
    async def update_llm_connection(
        connection_id: int,
        request: Request,
        name: str = Form(...),
        base_url: str = Form(...),
        model: str = Form(...),
        api_key_env_var: str = Form(default=""),
        timeout_seconds: int = Form(default=20),
        temperature: float = Form(default=0.2),
        max_output_tokens: int = Form(default=900),
        extra_headers_json: str = Form(default=""),
        is_enabled: bool = Form(default=False),
        is_default: bool = Form(default=False),
        csrf_token: str = Form(...),
        next_url: str = Form(default="/settings"),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        connection = session.get(LLMConnection, connection_id)
        if connection is None:
            raise HTTPException(status_code=404, detail="LLM connection not found")

        cleaned_name = name.strip()
        cleaned_base_url = base_url.strip().rstrip("/")
        cleaned_model = model.strip()
        cleaned_env_var = api_key_env_var.strip()
        if not cleaned_name:
            set_flash(request, "error", "Connection name is required.")
            return RedirectResponse(url=with_profile_id(next_url, connection.profile_id), status_code=303)
        if not cleaned_base_url.startswith("http://") and not cleaned_base_url.startswith("https://"):
            set_flash(request, "error", "LLM base URL must start with http:// or https://.")
            return RedirectResponse(url=with_profile_id(next_url, connection.profile_id), status_code=303)
        if not cleaned_model:
            set_flash(request, "error", "LLM model is required.")
            return RedirectResponse(url=with_profile_id(next_url, connection.profile_id), status_code=303)

        duplicate = session.exec(
            select(LLMConnection).where(
                LLMConnection.profile_id == connection.profile_id,
                LLMConnection.name == cleaned_name,
                LLMConnection.id != connection_id,
            )
        ).first()
        if duplicate is not None:
            set_flash(request, "error", f"LLM connection name '{cleaned_name}' is already in use.")
            return RedirectResponse(url=with_profile_id(next_url, connection.profile_id), status_code=303)

        try:
            parsed_headers = parse_extra_headers_json(extra_headers_json)
        except ValueError as exc:
            set_flash(request, "error", str(exc))
            return RedirectResponse(url=with_profile_id(next_url, connection.profile_id), status_code=303)

        normalized_enabled = bool(is_enabled)
        normalized_default = bool(is_default and normalized_enabled)
        if normalized_default:
            existing_defaults = session.exec(
                select(LLMConnection).where(
                    LLMConnection.profile_id == connection.profile_id,
                    LLMConnection.is_default.is_(True),
                    LLMConnection.id != connection_id,
                )
            ).all()
            for item in existing_defaults:
                item.is_default = False
                item.updated_at = utcnow()
                session.add(item)

        connection.name = cleaned_name
        connection.base_url = cleaned_base_url
        connection.model = cleaned_model
        connection.api_key_env_var = cleaned_env_var or None
        connection.timeout_seconds = max(1, min(timeout_seconds, 300))
        connection.temperature = max(0.0, min(float(temperature), 2.0))
        connection.max_output_tokens = max(1, min(max_output_tokens, 8192))
        connection.extra_headers_json = safe_json_dump(parsed_headers) if parsed_headers else None
        connection.is_enabled = normalized_enabled
        connection.is_default = normalized_default
        connection.updated_at = utcnow()
        session.add(connection)
        session.commit()

        set_flash(request, "success", f"LLM connection '{cleaned_name}' saved.")
        return RedirectResponse(url=with_profile_id(next_url, connection.profile_id), status_code=303)

    @app.post("/llm-connections/{connection_id}/test")
    async def test_llm_connection(
        connection_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/settings"),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        connection = session.get(LLMConnection, connection_id)
        if connection is None:
            raise HTTPException(status_code=404, detail="LLM connection not found")

        api_key = resolve_llm_api_key(connection)
        settings = build_llm_settings_from_connection(connection, api_key)
        client = OpenAICompatibleLLMClient(settings)
        try:
            await client.test_connection()
            set_flash(request, "success", f"LLM connection '{connection.name}' is reachable.")
        except LLMClientError as exc:
            set_flash(request, "error", f"LLM connection test failed: {exc}")

        return RedirectResponse(url=with_profile_id(next_url, connection.profile_id), status_code=303)

    @app.post("/llm-connections/{connection_id}/delete")
    async def delete_llm_connection(
        connection_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/settings"),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        connection = session.get(LLMConnection, connection_id)
        if connection is None:
            raise HTTPException(status_code=404, detail="LLM connection not found")

        profile_id_value = connection.profile_id
        run_ids = list(
            session.exec(select(SuggestionRun.id).where(SuggestionRun.llm_connection_id == connection_id)).all()
        )
        if run_ids:
            session.exec(delete(SuggestionProposal).where(SuggestionProposal.suggestion_run_id.in_(run_ids)))
            session.exec(delete(SuggestionAuditEvent).where(SuggestionAuditEvent.suggestion_run_id.in_(run_ids)))
        session.exec(delete(SuggestionRun).where(SuggestionRun.llm_connection_id == connection_id))
        session.delete(connection)
        session.commit()

        set_flash(request, "success", "LLM connection deleted.")
        return RedirectResponse(url=with_profile_id(next_url, profile_id_value), status_code=303)

    @app.post("/profiles/{profile_id}/test")
    async def test_profile_connection(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/settings"),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")
        if not require_enabled_profile(profile, request):
            active_profile = choose_active_profile(session, request, None)
            return RedirectResponse(
                url=with_profile_id(next_url, active_profile.id if active_profile else None),
                status_code=303,
            )

        token = resolve_profile_token(profile)
        if not token:
            set_flash(request, "error", "No token configured in profile or environment.")
            return RedirectResponse(
                url=with_profile_id(next_url, profile.id),
                status_code=303,
            )

        client = HAClient(
            base_url=profile.base_url,
            token=token,
            verify_tls=profile.verify_tls,
            timeout_seconds=profile.timeout_seconds,
        )

        try:
            result = await client.test_connection()
            ha_version = result.get("version", "unknown")
            set_flash(request, "success", f"Connection successful. Home Assistant version: {ha_version}")
        except HAClientError as exc:
            set_flash(request, "error", f"Connection failed: {exc}")

        return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)

    @app.post("/profiles/{profile_id}/sync")
    async def sync_profile(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/entities"),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")
        if not require_enabled_profile(profile, request):
            active_profile = choose_active_profile(session, request, None)
            return RedirectResponse(
                url=with_profile_id(next_url, active_profile.id if active_profile else None),
                status_code=303,
            )
        sync_run, sync_error = await perform_profile_sync(
            session,
            profile,
            request_id=getattr(request.state, "request_id", None),
        )
        if sync_error is not None:
            set_flash(request, "error", sync_error)
            return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
        if sync_run is None:
            set_flash(request, "error", "Sync failed: no run was created.")
            return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)

        set_flash(request, "success", f"Sync complete. Stored {sync_run.entity_count} entities.")
        return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)

    @app.post("/profiles/{profile_id}/sync-config")
    async def sync_profile_config(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/config-items"),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")
        if not require_enabled_profile(profile, request):
            active_profile = choose_active_profile(session, request, None)
            return RedirectResponse(
                url=with_profile_id(next_url, active_profile.id if active_profile else None),
                status_code=303,
            )

        token = resolve_profile_token(profile)
        if not token:
            set_flash(request, "error", "No token configured in profile or environment.")
            return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)

        client = HAClient(
            base_url=profile.base_url,
            token=token,
            verify_tls=profile.verify_tls,
            timeout_seconds=profile.timeout_seconds,
        )

        pulled_at = utcnow()
        start = perf_counter()
        profile_id_value = profile.id
        if profile_id_value is None:
            raise HTTPException(status_code=500, detail="Profile ID is unavailable")

        log_event(
            "config_sync_started",
            request_id=getattr(request.state, "request_id", None),
            profile_id=profile_id_value,
            profile_name=profile.name,
            pulled_at=pulled_at.isoformat(),
        )

        try:
            states = await client.fetch_states()
        except HAClientError as exc:
            failed_run = ConfigSyncRun(
                profile_id=profile_id_value,
                pulled_at=pulled_at,
                item_count=0,
                success_count=0,
                error_count=0,
                duration_ms=int((perf_counter() - start) * 1000),
                status="failed",
                error=str(exc),
            )
            session.add(failed_run)
            session.commit()

            set_flash(request, "error", f"Config sync failed: {exc}")
            log_event(
                "config_sync_failed",
                request_id=getattr(request.state, "request_id", None),
                profile_id=profile_id_value,
                profile_name=profile.name,
                error=str(exc),
            )
            return RedirectResponse(
                url=with_profile_id(next_url, profile_id_value),
                status_code=303,
            )

        registry_metadata: dict[str, list[dict[str, Any]]] = {}
        try:
            registry_metadata = await client.fetch_registry_metadata()
        except HAClientError as exc:
            log_event(
                "config_sync_registry_metadata_unavailable",
                request_id=getattr(request.state, "request_id", None),
                profile_id=profile_id_value,
                profile_name=profile.name,
                error=str(exc),
            )

        candidates = build_config_candidates(states, registry_metadata)

        config_sync_run = ConfigSyncRun(
            profile_id=profile_id_value,
            pulled_at=pulled_at,
            item_count=len(candidates),
            success_count=0,
            error_count=0,
            duration_ms=0,
            status="success",
            error=None,
        )
        session.add(config_sync_run)
        session.commit()
        session.refresh(config_sync_run)
        config_sync_run_id = config_sync_run.id
        if config_sync_run_id is None:
            raise HTTPException(status_code=500, detail="Config sync run ID is unavailable")

        snapshots: list[ConfigSnapshot] = []
        if candidates:
            semaphore = asyncio.Semaphore(CONFIG_SYNC_CONCURRENCY)
            snapshots = await asyncio.gather(
                *[
                    build_config_snapshot_from_candidate(
                        client=client,
                        semaphore=semaphore,
                        profile_id=profile_id_value,
                        config_sync_run_id=config_sync_run_id,
                        pulled_at=pulled_at,
                        candidate=candidate,
                    )
                    for candidate in candidates
                ]
            )

        success_count = sum(1 for snapshot in snapshots if snapshot.fetch_status == "success")
        error_count = len(snapshots) - success_count
        if error_count == 0:
            final_status = "success"
            final_error = None
        elif success_count > 0:
            final_status = "partial"
            final_error = None
        else:
            final_status = "failed"
            final_error = "No config items were fetched successfully."

        config_sync_run.item_count = len(snapshots)
        config_sync_run.success_count = success_count
        config_sync_run.error_count = error_count
        config_sync_run.duration_ms = int((perf_counter() - start) * 1000)
        config_sync_run.status = final_status
        config_sync_run.error = final_error

        session.add_all(snapshots)
        session.add(config_sync_run)
        session.commit()

        if final_status == "success":
            set_flash(
                request,
                "success",
                f"Config sync complete. Stored {success_count} config items.",
            )
            log_event(
                "config_sync_completed",
                request_id=getattr(request.state, "request_id", None),
                profile_id=profile_id_value,
                profile_name=profile.name,
                config_sync_run_id=config_sync_run_id,
                item_count=len(snapshots),
                success_count=success_count,
                error_count=error_count,
                duration_ms=config_sync_run.duration_ms,
                pulled_at=pulled_at.isoformat(),
            )
        elif final_status == "partial":
            set_flash(
                request,
                "success",
                "Config sync finished with partial results. "
                f"Stored {success_count} items and {error_count} errors.",
            )
            log_event(
                "config_sync_partial",
                request_id=getattr(request.state, "request_id", None),
                profile_id=profile_id_value,
                profile_name=profile.name,
                config_sync_run_id=config_sync_run_id,
                item_count=len(snapshots),
                success_count=success_count,
                error_count=error_count,
                duration_ms=config_sync_run.duration_ms,
                pulled_at=pulled_at.isoformat(),
            )
        else:
            set_flash(
                request,
                "error",
                "Config sync failed. No config items were fetched successfully.",
            )
            log_event(
                "config_sync_failed",
                request_id=getattr(request.state, "request_id", None),
                profile_id=profile_id_value,
                profile_name=profile.name,
                config_sync_run_id=config_sync_run_id,
                item_count=len(snapshots),
                success_count=success_count,
                error_count=error_count,
                duration_ms=config_sync_run.duration_ms,
                error=final_error,
                pulled_at=pulled_at.isoformat(),
            )

        return RedirectResponse(url=with_profile_id(next_url, profile_id_value), status_code=303)

    @app.post("/profiles/{profile_id}/suggestions/runs")
    async def queue_automation_suggestions_run(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/suggestions"),
        llm_connection_id: str = Form(default=""),
        config_sync_run_id: str = Form(default=""),
        snapshot_id: str = Form(default=""),
        q: str = Form(default=""),
        max_targets: int = Form(default=SUGGESTION_MAX_TARGETS_PER_RUN),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")
        if not require_enabled_profile(profile, request):
            active_profile = choose_active_profile(session, request, None)
            return RedirectResponse(
                url=with_profile_id(next_url, active_profile.id if active_profile else None),
                status_code=303,
            )

        selected_connection: LLMConnection | None = None
        normalized_connection_id = llm_connection_id.strip()
        if normalized_connection_id:
            try:
                parsed_connection_id = int(normalized_connection_id)
            except ValueError:
                set_flash(request, "error", "Invalid LLM connection selection.")
                return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
            candidate = session.get(LLMConnection, parsed_connection_id)
            if (
                candidate is not None
                and candidate.profile_id == profile.id
                and candidate.provider_kind in LLM_PROVIDER_KINDS
                and candidate.is_enabled
            ):
                selected_connection = candidate
        if selected_connection is None:
            selected_connection = get_default_llm_connection(session, profile.id)
        if selected_connection is None:
            set_flash(request, "error", "No enabled LLM connection configured for this profile.")
            return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)

        resolved_config_sync_run_id: int | None = None
        normalized_config_sync_run_id = config_sync_run_id.strip()
        if normalized_config_sync_run_id:
            try:
                parsed_config_sync_run_id = int(normalized_config_sync_run_id)
            except ValueError:
                set_flash(request, "error", "Invalid config sync run selection.")
                return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
            candidate_run = session.get(ConfigSyncRun, parsed_config_sync_run_id)
            if candidate_run is None or candidate_run.profile_id != profile.id:
                set_flash(request, "error", "Config sync run not found for this profile.")
                return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
            resolved_config_sync_run_id = candidate_run.id
        else:
            latest_config_sync_run = get_latest_config_sync_run(session, profile.id)
            if latest_config_sync_run is not None:
                resolved_config_sync_run_id = latest_config_sync_run.id

        normalized_snapshot_id = snapshot_id.strip()
        snapshot_ids: list[int] = []
        if normalized_snapshot_id:
            try:
                parsed_snapshot_id = int(normalized_snapshot_id)
            except ValueError:
                set_flash(request, "error", "Invalid config snapshot selection.")
                return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
            snapshot = session.get(ConfigSnapshot, parsed_snapshot_id)
            if (
                snapshot is None
                or snapshot.profile_id != profile.id
                or snapshot.kind != "automation"
                or snapshot.fetch_status != "success"
            ):
                set_flash(request, "error", "Selected config snapshot is not eligible for automation suggestions.")
                return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
            if snapshot.id is None:
                set_flash(request, "error", "Selected config snapshot has no identifier.")
                return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
            snapshot_ids = [snapshot.id]
            resolved_config_sync_run_id = snapshot.config_sync_run_id

        run_filters = {
            "q": q.strip(),
            "max_targets": max(1, min(max_targets, SUGGESTION_MAX_TARGETS_PER_RUN)),
            "snapshot_ids": snapshot_ids,
        }

        run = SuggestionRun(
            profile_id=profile.id,
            llm_connection_id=selected_connection.id if selected_connection.id is not None else 0,
            config_sync_run_id=resolved_config_sync_run_id,
            status="queued",
            target_count=0,
            processed_count=0,
            success_count=0,
            invalid_count=0,
            error_count=0,
            error=None,
            context_hash=None,
            filters_json=safe_json_dump(run_filters),
            result_summary_json=None,
            started_at=None,
            finished_at=None,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        create_suggestion_audit_event(
            session,
            profile_id=profile.id,
            suggestion_run_id=run.id,
            event_type="suggestion_run_queued",
            actor="user",
            payload={
                "llm_connection_id": selected_connection.id,
                "config_sync_run_id": resolved_config_sync_run_id,
                "snapshot_ids": snapshot_ids,
            },
        )
        session.commit()

        if SUGGESTION_WORKER is None:
            run.status = "failed"
            run.error = "suggestion_worker_unavailable"
            run.finished_at = utcnow()
            run.updated_at = utcnow()
            session.add(run)
            create_suggestion_audit_event(
                session,
                profile_id=profile.id,
                suggestion_run_id=run.id,
                event_type="suggestion_run_failed",
                actor="system",
                payload={"reason": run.error},
            )
            session.commit()
            set_flash(request, "error", "Suggestion worker is unavailable.")
            return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)

        if run.id is not None:
            SUGGESTION_WORKER.enqueue(run.id)
        set_flash(request, "success", f"Queued automation suggestion run #{run.id}.")

        return RedirectResponse(
            url=f"/suggestions/{run.id}?{build_query(profile_id=profile.id)}",
            status_code=303,
        )

    @app.get("/suggestions", response_class=HTMLResponse)
    async def list_suggestions(
        request: Request,
        profile_id: int | None = Query(default=None),
        status: str = Query(default=""),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        profile_count = int(session.exec(select(func.count()).select_from(Profile)).one())

        runs: list[SuggestionRun] = []
        active_connection: LLMConnection | None = None
        if active_profile is not None:
            active_connection = get_default_llm_connection(session, active_profile.id)
            stmt = select(SuggestionRun).where(SuggestionRun.profile_id == active_profile.id)
            normalized_status = status.strip().lower()
            if normalized_status in SUGGESTION_RUN_STATUSES:
                stmt = stmt.where(SuggestionRun.status == normalized_status)
            runs = list(session.exec(stmt.order_by(SuggestionRun.created_at.desc(), SuggestionRun.id.desc())).all())

        return render_template(
            request,
            "suggestions.html",
            with_navigation(
                request,
                session,
                {
                    "active_profile": active_profile,
                    "active_connection": active_connection,
                    "runs": runs,
                    "status": status.strip().lower(),
                    "profile_count": profile_count,
                },
                active_profile,
            ),
        )

    @app.get("/suggestions/{run_id}", response_class=HTMLResponse)
    async def suggestion_run_detail(
        run_id: int,
        request: Request,
        profile_id: int | None = Query(default=None),
        proposal_status: str = Query(default=""),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No enabled profiles found")

        run = session.get(SuggestionRun, run_id)
        if run is None or run.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Suggestion run not found")

        normalized_status = proposal_status.strip().lower()
        stmt = select(SuggestionProposal).where(SuggestionProposal.suggestion_run_id == run.id)
        if normalized_status and normalized_status in SUGGESTION_PROPOSAL_STATUSES:
            stmt = stmt.where(SuggestionProposal.status == normalized_status)
        proposals = list(session.exec(stmt.order_by(SuggestionProposal.created_at.asc(), SuggestionProposal.id.asc())).all())

        connection = session.get(LLMConnection, run.llm_connection_id)
        summary_payload = safe_json_load(run.result_summary_json, {})
        summary_pretty = safe_pretty_json(safe_json_dump(summary_payload))

        return render_template(
            request,
            "suggestion_run_detail.html",
            with_navigation(
                request,
                session,
                {
                    "active_profile": active_profile,
                    "run": run,
                    "connection": connection,
                    "proposals": proposals,
                    "proposal_status": normalized_status,
                    "summary_payload": summary_payload,
                    "summary_pretty": summary_pretty,
                    "back_url": f"/suggestions?{build_query(profile_id=active_profile.id)}",
                    "status_choices": sorted(SUGGESTION_PROPOSAL_STATUSES),
                },
                active_profile,
            ),
        )

    @app.post("/suggestions/proposals/{proposal_id}/status")
    async def update_suggestion_proposal_status(
        proposal_id: int,
        request: Request,
        status: str = Form(...),
        next_url: str = Form(default="/suggestions"),
        csrf_token: str = Form(...),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        proposal = session.get(SuggestionProposal, proposal_id)
        if proposal is None:
            raise HTTPException(status_code=404, detail="Suggestion proposal not found")

        normalized_status = status.strip().lower()
        if normalized_status not in {"accepted", "rejected"}:
            set_flash(request, "error", "Invalid proposal status.")
            return RedirectResponse(url=with_profile_id(next_url, proposal.profile_id), status_code=303)

        previous_status = proposal.status
        proposal.status = normalized_status
        proposal.updated_at = utcnow()
        session.add(proposal)
        create_suggestion_audit_event(
            session,
            profile_id=proposal.profile_id,
            suggestion_run_id=proposal.suggestion_run_id,
            proposal_id=proposal.id,
            event_type="suggestion_proposal_status_changed",
            actor="user",
            payload={"from": previous_status, "to": normalized_status},
        )
        session.commit()

        set_flash(request, "success", f"Suggestion marked as {normalized_status}.")
        return RedirectResponse(url=with_profile_id(next_url, proposal.profile_id), status_code=303)

    @app.get("/api/suggestions/runs/{run_id}")
    async def api_suggestion_run_status(
        run_id: int,
        request: Request,
        profile_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> JSONResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No enabled profile found")

        run = session.get(SuggestionRun, run_id)
        if run is None or run.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Suggestion run not found")

        proposals = list(
            session.exec(
                select(SuggestionProposal).where(SuggestionProposal.suggestion_run_id == run.id)
            ).all()
        )
        proposal_counts: dict[str, int] = {}
        for item in proposals:
            proposal_counts[item.status] = proposal_counts.get(item.status, 0) + 1

        payload = {
            "run": {
                "id": run.id,
                "profile_id": run.profile_id,
                "status": run.status,
                "target_count": run.target_count,
                "processed_count": run.processed_count,
                "success_count": run.success_count,
                "invalid_count": run.invalid_count,
                "error_count": run.error_count,
                "error": run.error,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "created_at": run.created_at.isoformat(),
                "updated_at": run.updated_at.isoformat(),
                "context_hash": run.context_hash,
                "summary": safe_json_load(run.result_summary_json, {}),
            },
            "proposal_counts": proposal_counts,
            "proposals": [
                {
                    "id": item.id,
                    "config_snapshot_id": item.config_snapshot_id,
                    "target_entity_id": item.target_entity_id,
                    "status": item.status,
                    "summary": item.summary,
                    "confidence": item.confidence,
                    "risk_level": item.risk_level,
                    "validation_error": item.validation_error,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                }
                for item in proposals
            ],
        }
        return JSONResponse(payload)

    @app.get("/entities", response_class=HTMLResponse)
    async def list_entities(
        request: Request,
        profile_id: int | None = Query(default=None),
        sync_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        domain: str = Query(default=""),
        state_value: str = Query(default="", alias="state"),
        changed_within: int | None = Query(default=None, ge=1, le=10080),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        profile_count = int(session.exec(select(func.count()).select_from(Profile)).one())

        active_sync_run: SyncRun | None = None
        if active_profile is not None:
            if sync_run_id is not None:
                candidate = session.get(SyncRun, sync_run_id)
                if candidate is not None and candidate.profile_id == active_profile.id:
                    active_sync_run = candidate
            if active_sync_run is None:
                active_sync_run = get_latest_sync_run(session, active_profile.id)

        entities: list[EntitySnapshot] = []
        total = 0
        total_pages = 1

        if active_profile is not None and active_sync_run is not None:
            filtered_stmt = build_entity_stmt(
                profile_id=active_profile.id,
                sync_run_id=active_sync_run.id,
                q=q,
                domain=domain,
                state_value=state_value,
                changed_within=changed_within,
            )

            count_stmt = select(func.count()).select_from(filtered_stmt.subquery())
            total = int(session.exec(count_stmt).one())

            total_pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, total_pages)
            offset = (page - 1) * page_size
            entities = session.exec(
                filtered_stmt.order_by(EntitySnapshot.entity_id).offset(offset).limit(page_size)
            ).all()

        domains: list[str] = []
        states: list[str] = []
        if active_profile is not None and active_sync_run is not None:
            domains = list(
                session.exec(
                select(EntitySnapshot.domain)
                .where(
                    EntitySnapshot.profile_id == active_profile.id,
                    EntitySnapshot.sync_run_id == active_sync_run.id,
                )
                .distinct()
                .order_by(EntitySnapshot.domain)
            ).all()
            )
            states = list(
                session.exec(
                select(EntitySnapshot.state)
                .where(
                    EntitySnapshot.profile_id == active_profile.id,
                    EntitySnapshot.sync_run_id == active_sync_run.id,
                )
                .distinct()
                .order_by(EntitySnapshot.state)
            ).all()
            )

        next_url_query = build_query(
            profile_id=active_profile.id if active_profile else None,
            sync_run_id=active_sync_run.id if active_sync_run else None,
            q=q,
            domain=domain,
            state=state_value,
            changed_within=changed_within,
            page=page,
            page_size=page_size,
        )
        next_url = f"/entities?{next_url_query}" if next_url_query else "/entities"

        return render_template(
            request,
            "entities.html",
            with_navigation(
                request,
                session,
                {
                    "active_profile": active_profile,
                    "active_sync_run": active_sync_run,
                    "entities": entities,
                    "domains": domains,
                    "states": states,
                    "q": q,
                    "domain": domain,
                    "state_value": state_value,
                    "changed_within": changed_within,
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages,
                    "next_url": next_url,
                    "profile_count": profile_count,
                    "profiles": get_enabled_profiles(session),
                },
                active_profile,
            ),
        )

    @app.get("/entities/{entity_id}", response_class=HTMLResponse)
    async def entity_detail(
        entity_id: str,
        request: Request,
        profile_id: int | None = Query(default=None),
        sync_run_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No profiles configured")

        active_sync_run: SyncRun | None = None
        if sync_run_id is not None:
            candidate = session.get(SyncRun, sync_run_id)
            if candidate is not None and candidate.profile_id == active_profile.id:
                active_sync_run = candidate
        if active_sync_run is None:
            active_sync_run = get_latest_sync_run(session, active_profile.id)
        if active_sync_run is None:
            raise HTTPException(status_code=404, detail="No synced entities found")

        snapshot = session.exec(
            select(EntitySnapshot)
            .where(
                EntitySnapshot.profile_id == active_profile.id,
                EntitySnapshot.sync_run_id == active_sync_run.id,
                EntitySnapshot.entity_id == entity_id,
            )
            .limit(1)
        ).first()
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        back_query = build_query(
            profile_id=active_profile.id,
            sync_run_id=active_sync_run.id,
        )

        return render_template(
            request,
            "entity_detail.html",
            with_navigation(
                request,
                session,
                {
                    "profile": active_profile,
                    "sync_run": active_sync_run,
                    "snapshot": snapshot,
                    "attributes_pretty": safe_pretty_json(snapshot.attributes_json),
                    "context_pretty": safe_pretty_json(snapshot.context_json),
                    "labels_pretty": safe_pretty_json(snapshot.labels_json),
                    "metadata_pretty": safe_pretty_json(snapshot.metadata_json),
                    "back_url": f"/entities?{back_query}",
                },
                active_profile,
            ),
        )

    @app.get("/config-items", response_class=HTMLResponse)
    async def list_config_items(
        request: Request,
        profile_id: int | None = Query(default=None),
        config_sync_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        kind: str = Query(default=""),
        status: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        normalized_kind = kind.strip()
        if normalized_kind and normalized_kind not in CONFIG_KINDS:
            normalized_kind = ""

        normalized_status = status.strip()
        if normalized_status and normalized_status not in {"success", "error"}:
            normalized_status = ""

        active_profile = choose_active_profile(session, request, profile_id)
        profile_count = int(session.exec(select(func.count()).select_from(Profile)).one())
        llm_connections: list[LLMConnection] = []
        active_llm_connection: LLMConnection | None = None

        config_runs: list[ConfigSyncRun] = []
        active_config_sync_run: ConfigSyncRun | None = None
        if active_profile is not None:
            llm_connections = get_profile_llm_connections(session, active_profile.id, enabled_only=True)
            active_llm_connection = llm_connections[0] if llm_connections else None
            config_runs = list(
                session.exec(
                    select(ConfigSyncRun)
                    .where(ConfigSyncRun.profile_id == active_profile.id)
                    .order_by(ConfigSyncRun.pulled_at.desc(), ConfigSyncRun.id.desc())
                ).all()
            )
            if config_sync_run_id is not None:
                candidate = session.get(ConfigSyncRun, config_sync_run_id)
                if candidate is not None and candidate.profile_id == active_profile.id:
                    active_config_sync_run = candidate
            if active_config_sync_run is None:
                active_config_sync_run = get_latest_config_sync_run(session, active_profile.id)

        config_items: list[ConfigSnapshot] = []
        total = 0
        total_pages = 1
        kinds: list[str] = []
        statuses: list[str] = []

        if active_profile is not None and active_config_sync_run is not None:
            filtered_stmt = build_config_item_stmt(
                profile_id=active_profile.id,
                config_sync_run_id=active_config_sync_run.id,
                q=q,
                kind=normalized_kind,
                status=normalized_status,
            )

            count_stmt = select(func.count()).select_from(filtered_stmt.subquery())
            total = int(session.exec(count_stmt).one())

            total_pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, total_pages)
            offset = (page - 1) * page_size
            config_items = session.exec(
                filtered_stmt.order_by(ConfigSnapshot.kind, ConfigSnapshot.entity_id)
                .offset(offset)
                .limit(page_size)
            ).all()

            kinds = list(
                session.exec(
                    select(ConfigSnapshot.kind)
                    .where(
                        ConfigSnapshot.profile_id == active_profile.id,
                        ConfigSnapshot.config_sync_run_id == active_config_sync_run.id,
                    )
                    .distinct()
                    .order_by(ConfigSnapshot.kind)
                ).all()
            )
            statuses = list(
                session.exec(
                    select(ConfigSnapshot.fetch_status)
                    .where(
                        ConfigSnapshot.profile_id == active_profile.id,
                        ConfigSnapshot.config_sync_run_id == active_config_sync_run.id,
                    )
                    .distinct()
                    .order_by(ConfigSnapshot.fetch_status)
                ).all()
            )

        next_url_query = build_query(
            profile_id=active_profile.id if active_profile else None,
            config_sync_run_id=active_config_sync_run.id if active_config_sync_run else None,
            q=q,
            kind=normalized_kind,
            status=normalized_status,
            page=page,
            page_size=page_size,
        )
        next_url = f"/config-items?{next_url_query}" if next_url_query else "/config-items"

        return render_template(
            request,
            "config_items.html",
            with_navigation(
                request,
                session,
                {
                    "active_profile": active_profile,
                    "config_runs": config_runs,
                    "active_config_sync_run": active_config_sync_run,
                    "config_items": config_items,
                    "kinds": kinds,
                    "statuses": statuses,
                    "q": q,
                    "kind": normalized_kind,
                    "status": normalized_status,
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages,
                    "next_url": next_url,
                    "profile_count": profile_count,
                    "profiles": get_enabled_profiles(session),
                    "llm_connections": llm_connections,
                    "active_llm_connection": active_llm_connection,
                },
                active_profile,
            ),
        )

    @app.get("/config-items/{snapshot_id}", response_class=HTMLResponse)
    async def config_item_detail(
        snapshot_id: int,
        request: Request,
        profile_id: int | None = Query(default=None),
        config_sync_run_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No enabled profiles found")

        snapshot = session.get(ConfigSnapshot, snapshot_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Config item not found")
        if snapshot.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Config item not found in requested profile")

        if config_sync_run_id is not None and snapshot.config_sync_run_id != config_sync_run_id:
            raise HTTPException(status_code=404, detail="Config item not found in requested sync run")

        profile = session.get(Profile, snapshot.profile_id)
        sync_run = session.get(ConfigSyncRun, snapshot.config_sync_run_id)
        if profile is None or sync_run is None or not profile.is_enabled:
            raise HTTPException(status_code=404, detail="Related profile or sync run not found")

        back_query = build_query(
            profile_id=snapshot.profile_id,
            config_sync_run_id=snapshot.config_sync_run_id,
        )

        return render_template(
            request,
            "config_item_detail.html",
            with_navigation(
                request,
                session,
                {
                    "profile": profile,
                    "sync_run": sync_run,
                    "snapshot": snapshot,
                    "summary_pretty": safe_pretty_json(snapshot.summary_json),
                    "references_pretty": safe_pretty_json(snapshot.references_json),
                    "config_pretty": safe_pretty_json(snapshot.config_json),
                    "attributes_pretty": safe_pretty_json(snapshot.attributes_json),
                    "metadata_pretty": safe_pretty_json(snapshot.metadata_json),
                    "back_url": f"/config-items?{back_query}",
                    "active_llm_connection": get_default_llm_connection(session, active_profile.id),
                },
                active_profile,
            ),
        )

    @app.post("/profiles/{profile_id}/run-entity-suggestions")
    async def run_entity_suggestions(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/entity-suggestions"),
        sync_run_id: str = Form(default=""),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")
        if not require_enabled_profile(profile, request):
            active_profile = choose_active_profile(session, request, None)
            return RedirectResponse(
                url=with_profile_id(next_url, active_profile.id if active_profile else None),
                status_code=303,
            )

        requested_sync_run: SyncRun | None = None
        normalized_sync_run_id = sync_run_id.strip()
        if normalized_sync_run_id:
            try:
                parsed_sync_run_id = int(normalized_sync_run_id)
            except ValueError:
                set_flash(request, "error", "Invalid sync run selection for suggestions.")
                return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
            candidate_sync_run = session.get(SyncRun, parsed_sync_run_id)
            if candidate_sync_run is not None and candidate_sync_run.profile_id == profile.id:
                requested_sync_run = candidate_sync_run

        active_sync_run = requested_sync_run or get_latest_sync_run(session, profile.id)
        if active_sync_run is None:
            set_flash(request, "error", "No synced entities found. Run entity sync first.")
            return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
        run, area_required, llm_error_count = await perform_entity_suggestion_run(
            session,
            profile,
            active_sync_run,
            request_id=getattr(request.state, "request_id", None),
        )

        area_downgrade_note = (
            " Area checks were downgraded because no area/device enrichment was available for this run."
            if not area_required
            else ""
        )
        if run.status == "success":
            set_flash(
                request,
                "success",
                "Suggestion check complete. "
                f"{run.ready_count} ready, {run.needs_review_count} needs review, {run.blocked_count} blocked."
                f"{area_downgrade_note}",
            )
        else:
            set_flash(
                request,
                "success",
                "Suggestion check finished with partial LLM results. "
                f"{run.ready_count} ready, {run.needs_review_count} needs review, {run.blocked_count} blocked."
                f"{area_downgrade_note} LLM failures: {llm_error_count}.",
            )

        next_target = with_profile_id(next_url, profile.id)
        split_result = urlsplit(next_target)
        query_items = [
            (k, v)
            for k, v in parse_qsl(split_result.query, keep_blank_values=True)
            if k != "suggestion_run_id"
        ]
        query_items.append(("suggestion_run_id", str(run.id)))
        next_target = urlunsplit(
            ("", "", split_result.path or "/entity-suggestions", urlencode(query_items), "")
        )
        return RedirectResponse(url=next_target, status_code=303)

    @app.post("/profiles/{profile_id}/entity-suggestions/recheck")
    async def recheck_entity_suggestions(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/entity-suggestions/workflow"),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")
        if not require_enabled_profile(profile, request):
            active_profile = choose_active_profile(session, request, None)
            return RedirectResponse(
                url=with_profile_id(next_url, active_profile.id if active_profile else None),
                status_code=303,
            )

        request_id = getattr(request.state, "request_id", None)
        log_event(
            "entity_suggestion_recheck_started",
            request_id=request_id,
            profile_id=profile.id,
            profile_name=profile.name,
        )

        sync_run, sync_error = await perform_profile_sync(
            session,
            profile,
            request_id=request_id,
        )
        if sync_error is not None or sync_run is None:
            set_flash(
                request,
                "error",
                sync_error or "Recheck failed during sync.",
            )
            return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)

        run, area_required, llm_error_count = await perform_entity_suggestion_run(
            session,
            profile,
            sync_run,
            request_id=request_id,
        )
        area_downgrade_note = (
            " Area checks were downgraded because no area/device enrichment was available for this run."
            if not area_required
            else ""
        )
        if run.status == "success":
            set_flash(
                request,
                "success",
                "Recheck complete. "
                f"{run.ready_count} ready, {run.needs_review_count} needs review, {run.blocked_count} blocked."
                f"{area_downgrade_note}",
            )
        else:
            set_flash(
                request,
                "success",
                "Recheck complete with partial LLM results. "
                f"{run.ready_count} ready, {run.needs_review_count} needs review, {run.blocked_count} blocked."
                f"{area_downgrade_note} LLM failures: {llm_error_count}.",
            )

        log_event(
            "entity_suggestion_recheck_completed",
            request_id=request_id,
            profile_id=profile.id,
            profile_name=profile.name,
            sync_run_id=sync_run.id,
            suggestion_run_id=run.id,
            suggestion_status=run.status,
        )

        next_target = with_profile_id(next_url, profile.id)
        split_result = urlsplit(next_target)
        query_items = [
            (k, v)
            for k, v in parse_qsl(split_result.query, keep_blank_values=True)
            if k != "suggestion_run_id"
        ]
        if run.id is not None:
            query_items.append(("suggestion_run_id", str(run.id)))
        next_target = urlunsplit(
            ("", "", split_result.path or "/entity-suggestions/workflow", urlencode(query_items), "")
        )
        return RedirectResponse(url=next_target, status_code=303)

    @app.get("/entity-suggestions", response_class=HTMLResponse)
    async def list_entity_suggestions(
        request: Request,
        profile_id: int | None = Query(default=None),
        suggestion_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        status: str = Query(default=""),
        domain: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        normalized_status = status.strip()
        if normalized_status and normalized_status not in READINESS_STATUSES:
            normalized_status = ""

        active_profile = choose_active_profile(session, request, profile_id)
        profile_count = int(session.exec(select(func.count()).select_from(Profile)).one())

        suggestion_runs: list[EntitySuggestionRun] = []
        active_suggestion_run: EntitySuggestionRun | None = None
        if active_profile is not None:
            suggestion_runs = list(
                session.exec(
                    select(EntitySuggestionRun)
                    .where(EntitySuggestionRun.profile_id == active_profile.id)
                    .order_by(EntitySuggestionRun.pulled_at.desc(), EntitySuggestionRun.id.desc())
                ).all()
            )
            if suggestion_run_id is not None:
                candidate_run = session.get(EntitySuggestionRun, suggestion_run_id)
                if candidate_run is not None and candidate_run.profile_id == active_profile.id:
                    active_suggestion_run = candidate_run
            if active_suggestion_run is None:
                active_suggestion_run = get_latest_suggestion_run(session, active_profile.id)

        suggestions: list[EntitySuggestion] = []
        total = 0
        total_pages = 1
        statuses: list[str] = []
        domains: list[str] = []
        run_area_check_degraded = False
        suggestion_scope_domains_text = ", ".join(
            [item for item in ("sensor", "binary_sensor", "lock") if item in ACTIONABLE_SUGGESTION_DOMAINS]
        )
        if active_profile is not None and active_suggestion_run is not None:
            filtered_stmt = build_entity_suggestion_stmt(
                profile_id=active_profile.id,
                suggestion_run_id=active_suggestion_run.id,
                q=q,
                readiness_status=normalized_status,
                domain=domain,
            )
            count_stmt = select(func.count()).select_from(filtered_stmt.subquery())
            total = int(session.exec(count_stmt).one())

            total_pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, total_pages)
            offset = (page - 1) * page_size
            suggestions = list(
                session.exec(
                    filtered_stmt.order_by(EntitySuggestion.entity_id).offset(offset).limit(page_size)
                ).all()
            )

            statuses = list(
                session.exec(
                    select(EntitySuggestion.readiness_status)
                    .where(
                        EntitySuggestion.profile_id == active_profile.id,
                        EntitySuggestion.suggestion_run_id == active_suggestion_run.id,
                    )
                    .distinct()
                    .order_by(EntitySuggestion.readiness_status)
                ).all()
            )
            domains = list(
                session.exec(
                    select(EntitySuggestion.domain)
                    .where(
                        EntitySuggestion.profile_id == active_profile.id,
                        EntitySuggestion.suggestion_run_id == active_suggestion_run.id,
                    )
                    .distinct()
                    .order_by(EntitySuggestion.domain)
                ).all()
            )
            run_area_check_degraded = (
                int(
                    session.exec(
                        select(func.count())
                        .select_from(EntitySuggestion)
                        .where(
                            EntitySuggestion.profile_id == active_profile.id,
                            EntitySuggestion.suggestion_run_id == active_suggestion_run.id,
                            func.coalesce(EntitySuggestion.source_metadata_json, "").like(
                                '%"area_check_mode"%degraded_no_enrichment%'
                            ),
                        )
                    ).one()
                )
                > 0
            )

        next_url_query = build_query(
            profile_id=active_profile.id if active_profile else None,
            suggestion_run_id=active_suggestion_run.id if active_suggestion_run else None,
            q=q,
            status=normalized_status,
            domain=domain,
            page=page,
            page_size=page_size,
        )
        next_url = f"/entity-suggestions?{next_url_query}" if next_url_query else "/entity-suggestions"

        return render_template(
            request,
            "entity_suggestions.html",
            with_navigation(
                request,
                session,
                {
                    "active_profile": active_profile,
                    "active_suggestion_run": active_suggestion_run,
                    "suggestion_runs": suggestion_runs,
                    "suggestions": suggestions,
                    "q": q,
                    "status": normalized_status,
                    "domain": domain,
                    "statuses": statuses,
                    "domains": domains,
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages,
                    "next_url": next_url,
                    "profile_count": profile_count,
                    "profiles": get_enabled_profiles(session),
                    "suggestion_scope_domains_text": suggestion_scope_domains_text,
                    "run_area_check_degraded": run_area_check_degraded,
                },
                active_profile,
            ),
        )

    @app.get("/entity-suggestions/workflow", response_class=HTMLResponse)
    async def entity_suggestion_workflow_queue(
        request: Request,
        profile_id: int | None = Query(default=None),
        suggestion_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        domain: str = Query(default=""),
        workflow_status: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        normalized_workflow_status = workflow_status.strip()
        if normalized_workflow_status and normalized_workflow_status not in WORKFLOW_STATUSES:
            normalized_workflow_status = ""

        active_profile = choose_active_profile(session, request, profile_id)
        profile_count = int(session.exec(select(func.count()).select_from(Profile)).one())

        suggestion_runs: list[EntitySuggestionRun] = []
        active_suggestion_run: EntitySuggestionRun | None = None
        if active_profile is not None:
            suggestion_runs = list(
                session.exec(
                    select(EntitySuggestionRun)
                    .where(EntitySuggestionRun.profile_id == active_profile.id)
                    .order_by(EntitySuggestionRun.pulled_at.desc(), EntitySuggestionRun.id.desc())
                ).all()
            )
            if suggestion_run_id is not None:
                candidate_run = session.get(EntitySuggestionRun, suggestion_run_id)
                if candidate_run is not None and candidate_run.profile_id == active_profile.id:
                    active_suggestion_run = candidate_run
            if active_suggestion_run is None:
                active_suggestion_run = get_latest_suggestion_run(session, active_profile.id)

        queue_items: list[dict[str, Any]] = []
        paged_items: list[dict[str, Any]] = []
        workflow_counts = {status: 0 for status in sorted(WORKFLOW_STATUSES)}
        total = 0
        total_pages = 1
        domains: list[str] = []
        next_unresolved_id: int | None = None

        if active_profile is not None and active_suggestion_run is not None:
            base_stmt = select(EntitySuggestion).where(
                EntitySuggestion.profile_id == active_profile.id,
                EntitySuggestion.suggestion_run_id == active_suggestion_run.id,
                EntitySuggestion.readiness_status.in_(["blocked", "needs_review"]),
            )
            if q:
                pattern = f"%{q.lower()}%"
                base_stmt = base_stmt.where(
                    or_(
                        func.lower(EntitySuggestion.entity_id).like(pattern),
                        func.lower(func.coalesce(EntitySuggestion.issues_json, "")).like(pattern),
                        func.lower(func.coalesce(EntitySuggestion.missing_fields_json, "")).like(pattern),
                    )
                )
            if domain:
                base_stmt = base_stmt.where(EntitySuggestion.domain == domain)

            rows = list(session.exec(base_stmt).all())
            for row in rows:
                fixable_issues, manual_issues = build_workflow_issue_sections(row)
                if not fixable_issues:
                    continue
                queue_items.append(
                    {
                        "suggestion": row,
                        "fixable_issues": fixable_issues,
                        "manual_issues": manual_issues,
                        "fixable_count": len(fixable_issues),
                    }
                )

            for status_name in workflow_counts:
                workflow_counts[status_name] = sum(
                    1
                    for item in queue_items
                    if item["suggestion"].workflow_status == status_name
                )

            queue_items.sort(
                key=lambda item: (
                    workflow_status_rank(item["suggestion"].readiness_status),
                    -item["fixable_count"],
                    item["suggestion"].entity_id,
                )
            )

            for item in queue_items:
                suggestion_id_value = item["suggestion"].id
                if suggestion_id_value is None:
                    continue
                if item["suggestion"].workflow_status in {"open", "error"}:
                    next_unresolved_id = suggestion_id_value
                    break

            if normalized_workflow_status:
                queue_items = [
                    item
                    for item in queue_items
                    if item["suggestion"].workflow_status == normalized_workflow_status
                ]

            domains = sorted({item["suggestion"].domain for item in queue_items})

            total = len(queue_items)
            total_pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, total_pages)
            offset = (page - 1) * page_size
            paged_items = queue_items[offset : offset + page_size]

        next_url_query = build_query(
            profile_id=active_profile.id if active_profile else None,
            suggestion_run_id=active_suggestion_run.id if active_suggestion_run else None,
            q=q,
            domain=domain,
            workflow_status=normalized_workflow_status,
            page=page,
            page_size=page_size,
        )
        next_url = (
            f"/entity-suggestions/workflow?{next_url_query}"
            if next_url_query
            else "/entity-suggestions/workflow"
        )

        return render_template(
            request,
            "entity_suggestion_workflow_queue.html",
            with_navigation(
                request,
                session,
                {
                    "active_profile": active_profile,
                    "active_suggestion_run": active_suggestion_run,
                    "suggestion_runs": suggestion_runs,
                    "queue_items": paged_items,
                    "workflow_counts": workflow_counts,
                    "workflow_status": normalized_workflow_status,
                    "workflow_statuses": sorted(WORKFLOW_STATUSES),
                    "domains": domains,
                    "domain": domain,
                    "q": q,
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages,
                    "next_url": next_url,
                    "next_unresolved_id": next_unresolved_id,
                    "profile_count": profile_count,
                    "profiles": get_enabled_profiles(session),
                },
                active_profile,
            ),
        )

    @app.get("/entity-suggestions/{suggestion_id}/workflow", response_class=HTMLResponse)
    async def entity_suggestion_workflow_detail(
        suggestion_id: int,
        request: Request,
        profile_id: int | None = Query(default=None),
        suggestion_run_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No enabled profiles found")

        suggestion = session.get(EntitySuggestion, suggestion_id)
        if suggestion is None or suggestion.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Suggestion not found in requested profile")
        if suggestion_run_id is not None and suggestion.suggestion_run_id != suggestion_run_id:
            raise HTTPException(status_code=404, detail="Suggestion not found in requested run")

        run = session.get(EntitySuggestionRun, suggestion.suggestion_run_id)
        snapshot = session.get(EntitySnapshot, suggestion.entity_snapshot_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Suggestion run not found")

        fixable_issues, manual_issues = build_workflow_issue_sections(suggestion)
        issue_codes = issue_codes_for_suggestion(suggestion)
        allow_friendly_name = bool({"missing_friendly_name", "generic_friendly_name"} & issue_codes)
        allow_area = bool({"missing_area", "missing_area_enrichment_unavailable"} & issue_codes)
        allow_device_class = bool(
            {"missing_sensor_semantic_type", "missing_binary_sensor_device_class"} & issue_codes
        )
        allow_labels = "missing_labels" in issue_codes

        area_options_by_id: dict[str, dict[str, str]] = {}
        area_options_source = "snapshot"
        area_options_error: str | None = None

        area_rows = list(
            session.exec(
                select(EntitySnapshot.area_id, EntitySnapshot.area_name)
                .where(
                    EntitySnapshot.profile_id == active_profile.id,
                    EntitySnapshot.sync_run_id == run.sync_run_id,
                )
                .where(
                    or_(
                        EntitySnapshot.area_id.is_not(None),
                        EntitySnapshot.area_name.is_not(None),
                    )
                )
                .distinct()
            ).all()
        )
        for area_id_value, area_name_value in area_rows:
            area_id_clean = as_clean_string(area_id_value)
            area_name_clean = as_clean_string(area_name_value)
            if area_id_clean is None and area_name_clean is None:
                continue
            option_area_id = area_id_clean or area_name_clean or ""
            option_area_name = area_name_clean or area_id_clean or ""
            if not option_area_id:
                continue
            area_options_by_id[option_area_id] = {
                "area_id": option_area_id,
                "area_name": option_area_name,
            }

        if allow_area:
            token = resolve_profile_token(active_profile)
            if token:
                client = HAClient(
                    base_url=active_profile.base_url,
                    token=token,
                    verify_tls=active_profile.verify_tls,
                    timeout_seconds=active_profile.timeout_seconds,
                )
                try:
                    area_entries = await client.fetch_area_registry_entries()
                    area_options_by_id = {}
                    for area_item in area_entries:
                        area_id_clean = as_clean_string(area_item.get("area_id") or area_item.get("id"))
                        area_name_clean = as_clean_string(area_item.get("name"))
                        if area_id_clean is None and area_name_clean is None:
                            continue
                        option_area_id = area_id_clean or area_name_clean or ""
                        option_area_name = area_name_clean or area_id_clean or ""
                        if not option_area_id:
                            continue
                        area_options_by_id[option_area_id] = {
                            "area_id": option_area_id,
                            "area_name": option_area_name,
                        }
                    area_options_source = "home_assistant"
                except HAClientError as exc:
                    area_options_error = str(exc)
                    area_options_source = "snapshot"
            else:
                area_options_error = "No token configured in profile or environment."
                area_options_source = "snapshot"

        area_options = sorted(
            area_options_by_id.values(),
            key=lambda item: item["area_name"].lower(),
        )

        label_options_by_id: dict[str, str] = {}
        label_rows = list(
            session.exec(
                select(EntitySnapshot.labels_json).where(
                    EntitySnapshot.profile_id == active_profile.id,
                    EntitySnapshot.sync_run_id == run.sync_run_id,
                )
            ).all()
        )
        for raw_labels in label_rows:
            payload = safe_json_load(raw_labels, {})
            if not isinstance(payload, dict):
                continue
            ids = payload.get("ids")
            names = payload.get("names")
            if not isinstance(ids, list):
                continue
            names_list: list[str] = names if isinstance(names, list) else []
            for idx, label_id_raw in enumerate(ids):
                if not isinstance(label_id_raw, str):
                    continue
                label_id_clean = label_id_raw.strip()
                if not label_id_clean:
                    continue
                label_name = label_id_clean
                if idx < len(names_list) and isinstance(names_list[idx], str):
                    candidate_name = names_list[idx].strip()
                    if candidate_name:
                        label_name = candidate_name
                label_options_by_id[label_id_clean] = label_name
        label_options = [
            {"label_id": label_id, "label_name": label_name}
            for label_id, label_name in sorted(
                label_options_by_id.items(),
                key=lambda item: item[1].lower(),
            )
        ]

        current_label_ids = parse_label_ids_from_snapshot(snapshot)
        current_label_ids_csv = ",".join(current_label_ids)
        current_label_names = parse_label_names_from_snapshot(snapshot)

        metadata_payload = safe_json_load(snapshot.metadata_json if snapshot else None, {})
        attributes_payload = safe_json_load(snapshot.attributes_json if snapshot else None, {})
        current_device_class = as_clean_string(
            metadata_payload.get("attribute_device_class") if isinstance(metadata_payload, dict) else None
        ) or as_clean_string(
            attributes_payload.get("device_class") if isinstance(attributes_payload, dict) else None
        )

        queue_rows = list(
            session.exec(
                select(EntitySuggestion)
                .where(
                    EntitySuggestion.profile_id == active_profile.id,
                    EntitySuggestion.suggestion_run_id == run.id,
                    EntitySuggestion.readiness_status.in_(["blocked", "needs_review"]),
                )
                .order_by(EntitySuggestion.entity_id)
            ).all()
        )
        queue_items: list[dict[str, Any]] = []
        for row in queue_rows:
            fixable, _ = split_suggestion_issues(row)
            if not fixable:
                continue
            queue_items.append(
                {
                    "suggestion": row,
                    "fixable_count": len(fixable),
                }
            )
        queue_items.sort(
            key=lambda item: (
                workflow_status_rank(item["suggestion"].readiness_status),
                -item["fixable_count"],
                item["suggestion"].entity_id,
            )
        )

        next_unresolved_id: int | None = None
        if queue_items:
            current_index = next(
                (idx for idx, item in enumerate(queue_items) if item["suggestion"].id == suggestion.id),
                None,
            )
            if current_index is not None:
                for item in queue_items[current_index + 1 :]:
                    suggestion_id_value = item["suggestion"].id
                    if (
                        suggestion_id_value is not None
                        and item["suggestion"].workflow_status in {"open", "error"}
                    ):
                        next_unresolved_id = suggestion_id_value
                        break
            if next_unresolved_id is None:
                for item in queue_items:
                    suggestion_id_value = item["suggestion"].id
                    if (
                        suggestion_id_value is not None
                        and item["suggestion"].workflow_status in {"open", "error"}
                    ):
                        next_unresolved_id = suggestion_id_value
                        break

        back_url = (
            "/entity-suggestions/workflow?"
            + build_query(profile_id=active_profile.id, suggestion_run_id=run.id)
        )
        apply_next_url = (
            f"/entity-suggestions/{suggestion.id}/workflow?"
            + build_query(profile_id=active_profile.id, suggestion_run_id=run.id)
        )

        return render_template(
            request,
            "entity_suggestion_workflow_detail.html",
            with_navigation(
                request,
                session,
                {
                    "active_profile": active_profile,
                    "run": run,
                    "suggestion": suggestion,
                    "snapshot": snapshot,
                    "fixable_issues": fixable_issues,
                    "manual_issues": manual_issues,
                    "allow_friendly_name": allow_friendly_name,
                    "allow_area": allow_area,
                    "allow_device_class": allow_device_class,
                    "allow_labels": allow_labels,
                    "area_options": area_options,
                    "area_options_source": area_options_source,
                    "area_options_error": area_options_error,
                    "area_create_option_value": AREA_CREATE_OPTION_VALUE,
                    "label_options": label_options,
                    "current_device_class": current_device_class or "",
                    "current_label_ids": current_label_ids,
                    "current_label_ids_csv": current_label_ids_csv,
                    "current_label_names": current_label_names,
                    "next_unresolved_id": next_unresolved_id,
                    "back_url": back_url,
                    "apply_next_url": apply_next_url,
                },
                active_profile,
            ),
        )

    @app.post("/entity-suggestions/{suggestion_id}/workflow/apply")
    async def apply_entity_suggestion_workflow(
        suggestion_id: int,
        request: Request,
        csrf_token: str = Form(...),
        profile_id: str = Form(default=""),
        next_url: str = Form(default="/entity-suggestions/workflow"),
        friendly_name: str = Form(default=""),
        area_id: str = Form(default=""),
        new_area_name: str = Form(default=""),
        device_class: str = Form(default=""),
        labels_csv: str = Form(default=""),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        resolved_profile_id: int | None = None
        if profile_id.strip():
            try:
                resolved_profile_id = int(profile_id.strip())
            except ValueError:
                resolved_profile_id = None
        active_profile = choose_active_profile(session, request, resolved_profile_id)
        suggestion = session.get(EntitySuggestion, suggestion_id)
        if suggestion is None:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        if active_profile is None or suggestion.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Suggestion not found in active profile")

        snapshot = session.get(EntitySnapshot, suggestion.entity_snapshot_id)
        if snapshot is None:
            suggestion.workflow_status = "error"
            suggestion.workflow_error = "Entity snapshot not found for suggestion."
            suggestion.workflow_updated_at = utcnow()
            session.add(suggestion)
            session.commit()
            set_flash(request, "error", "Cannot apply changes: snapshot is missing.")
            return RedirectResponse(url=with_profile_id(next_url, active_profile.id), status_code=303)

        issue_codes = issue_codes_for_suggestion(suggestion)
        fixable_codes = {code for code in issue_codes if code in WORKFLOW_FIXABLE_ISSUE_CODES}
        allow_friendly_name = bool({"missing_friendly_name", "generic_friendly_name"} & fixable_codes)
        allow_area = bool({"missing_area", "missing_area_enrichment_unavailable"} & fixable_codes)
        allow_device_class = bool(
            {"missing_sensor_semantic_type", "missing_binary_sensor_device_class"} & fixable_codes
        )
        allow_labels = "missing_labels" in fixable_codes

        submitted_payload = {
            "friendly_name": friendly_name.strip(),
            "area_id": area_id.strip(),
            "new_area_name": new_area_name.strip(),
            "device_class": device_class.strip(),
            "labels_csv": labels_csv.strip(),
        }

        def persist_workflow_error(message: str) -> RedirectResponse:
            suggestion.workflow_status = "error"
            suggestion.workflow_error = message
            suggestion.workflow_payload_json = safe_json_dump(submitted_payload)
            suggestion.workflow_result_json = safe_json_dump({"error": message})
            suggestion.workflow_updated_at = utcnow()
            session.add(suggestion)
            session.commit()
            log_event(
                "entity_suggestion_workflow_apply_failed",
                request_id=getattr(request.state, "request_id", None),
                profile_id=active_profile.id,
                profile_name=active_profile.name,
                suggestion_id=suggestion.id,
                entity_id=suggestion.entity_id,
                error=message,
            )
            set_flash(request, "error", f"Could not apply changes: {message}")
            return RedirectResponse(url=with_profile_id(next_url, active_profile.id), status_code=303)

        if not fixable_codes:
            return persist_workflow_error(
                "This suggestion has no workflow-editable issues. Review it manually in Home Assistant."
            )

        new_friendly_name = as_clean_string(friendly_name)
        selected_area_id_raw = area_id.strip()
        selected_area_id = as_clean_string(selected_area_id_raw)
        create_new_area_requested = selected_area_id == AREA_CREATE_OPTION_VALUE
        if create_new_area_requested:
            selected_area_id = None
        new_area_name_clean = as_clean_string(new_area_name)
        new_device_class = as_clean_string(device_class)
        submitted_label_ids = normalize_label_ids_form(labels_csv.split(","))

        if (new_friendly_name is not None or friendly_name.strip()) and not allow_friendly_name:
            return persist_workflow_error("Friendly name is not editable for this suggestion.")
        if (selected_area_id or new_area_name_clean) and not allow_area:
            return persist_workflow_error("Area is not editable for this suggestion.")
        if (new_device_class is not None or device_class.strip()) and not allow_device_class:
            return persist_workflow_error("Device class is not editable for this suggestion.")
        if (submitted_label_ids or labels_csv.strip()) and not allow_labels:
            return persist_workflow_error("Labels are not editable for this suggestion.")
        if selected_area_id and new_area_name_clean:
            return persist_workflow_error("Choose either an existing area or a new area name, not both.")
        if create_new_area_requested and new_area_name_clean is None:
            return persist_workflow_error("Enter a new area name when using Create New Area.")

        metadata_payload = safe_json_load(snapshot.metadata_json, {})
        attributes_payload = safe_json_load(snapshot.attributes_json, {})
        current_device_class = as_clean_string(
            metadata_payload.get("attribute_device_class") if isinstance(metadata_payload, dict) else None
        ) or as_clean_string(
            attributes_payload.get("device_class") if isinstance(attributes_payload, dict) else None
        )
        current_label_ids = parse_label_ids_from_snapshot(snapshot)

        update_payload: dict[str, Any] = {}
        applied_fields: list[str] = []
        area_resolution: dict[str, Any] | None = None

        current_name = as_clean_string(snapshot.friendly_name)
        if allow_friendly_name and new_friendly_name is not None and new_friendly_name != current_name:
            update_payload["name"] = new_friendly_name
            applied_fields.append("friendly_name")

        if allow_device_class and new_device_class is not None and new_device_class != current_device_class:
            update_payload["device_class"] = new_device_class
            applied_fields.append("device_class")

        if allow_labels and submitted_label_ids and submitted_label_ids != current_label_ids:
            update_payload["labels"] = submitted_label_ids
            applied_fields.append("labels")

        token = resolve_profile_token(active_profile)
        if not token:
            return persist_workflow_error("No token configured in profile or environment.")
        client = HAClient(
            base_url=active_profile.base_url,
            token=token,
            verify_tls=active_profile.verify_tls,
            timeout_seconds=active_profile.timeout_seconds,
        )

        try:
            resolved_area_id: str | None = None
            if allow_area and (selected_area_id or new_area_name_clean):
                if selected_area_id:
                    resolved_area_id = selected_area_id
                    area_resolution = {"mode": "existing", "area_id": selected_area_id}
                else:
                    area_entries = await client.fetch_area_registry_entries()
                    existing_area_id: str | None = None
                    for area_item in area_entries:
                        raw_area_name = as_clean_string(area_item.get("name"))
                        raw_area_id = as_clean_string(area_item.get("area_id") or area_item.get("id"))
                        if (
                            raw_area_name is not None
                            and new_area_name_clean is not None
                            and raw_area_name.casefold() == new_area_name_clean.casefold()
                            and raw_area_id is not None
                        ):
                            existing_area_id = raw_area_id
                            break
                    if existing_area_id is not None:
                        resolved_area_id = existing_area_id
                        area_resolution = {
                            "mode": "existing_by_name",
                            "area_name": new_area_name_clean,
                            "area_id": existing_area_id,
                        }
                    else:
                        created = await client.create_area_registry_entry(name=new_area_name_clean or "")
                        resolved_area_id = as_clean_string(created.get("area_id"))
                        area_resolution = {
                            "mode": "created",
                            "area_name": new_area_name_clean,
                            "area_id": resolved_area_id,
                        }
                current_area_id = as_clean_string(snapshot.area_id)
                if resolved_area_id is not None and resolved_area_id != current_area_id:
                    update_payload["area_id"] = resolved_area_id
                    applied_fields.append("area")

            if not update_payload:
                return persist_workflow_error("No applicable changes were provided for this suggestion.")

            log_event(
                "entity_suggestion_workflow_apply_started",
                request_id=getattr(request.state, "request_id", None),
                profile_id=active_profile.id,
                profile_name=active_profile.name,
                suggestion_id=suggestion.id,
                entity_id=suggestion.entity_id,
                fields=sorted(update_payload.keys()),
            )
            entity_entry = await client.update_entity_registry_entry(
                entity_id=snapshot.entity_id,
                **update_payload,
            )
        except HAClientError as exc:
            return persist_workflow_error(str(exc))

        suggestion.workflow_status = "applied_pending_recheck"
        suggestion.workflow_error = None
        suggestion.workflow_payload_json = safe_json_dump(submitted_payload)
        suggestion.workflow_result_json = safe_json_dump(
            {
                "applied_fields": applied_fields,
                "entity_update": entity_entry,
                "area_resolution": area_resolution,
            }
        )
        suggestion.workflow_updated_at = utcnow()
        session.add(suggestion)
        session.commit()

        log_event(
            "entity_suggestion_workflow_apply_succeeded",
            request_id=getattr(request.state, "request_id", None),
            profile_id=active_profile.id,
            profile_name=active_profile.name,
            suggestion_id=suggestion.id,
            entity_id=suggestion.entity_id,
            fields=applied_fields,
        )
        set_flash(
            request,
            "success",
            f"Applied updates for {suggestion.entity_id}. Run batch recheck after finishing your queue.",
        )
        return RedirectResponse(url=with_profile_id(next_url, active_profile.id), status_code=303)

    @app.post("/entity-suggestions/{suggestion_id}/workflow/skip")
    async def skip_entity_suggestion_workflow(
        suggestion_id: int,
        request: Request,
        csrf_token: str = Form(...),
        profile_id: str = Form(default=""),
        next_url: str = Form(default="/entity-suggestions/workflow"),
        reason: str = Form(default=""),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        resolved_profile_id: int | None = None
        if profile_id.strip():
            try:
                resolved_profile_id = int(profile_id.strip())
            except ValueError:
                resolved_profile_id = None
        active_profile = choose_active_profile(session, request, resolved_profile_id)
        suggestion = session.get(EntitySuggestion, suggestion_id)
        if suggestion is None:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        if active_profile is None or suggestion.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Suggestion not found in active profile")

        suggestion.workflow_status = "skipped"
        suggestion.workflow_error = None
        suggestion.workflow_payload_json = safe_json_dump({"reason": reason.strip() or None})
        suggestion.workflow_result_json = safe_json_dump({"status": "skipped"})
        suggestion.workflow_updated_at = utcnow()
        session.add(suggestion)
        session.commit()

        set_flash(request, "success", f"Skipped {suggestion.entity_id}.")
        return RedirectResponse(url=with_profile_id(next_url, active_profile.id), status_code=303)

    @app.get("/entity-suggestions/{suggestion_id}", response_class=HTMLResponse)
    async def entity_suggestion_detail(
        suggestion_id: int,
        request: Request,
        profile_id: int | None = Query(default=None),
        suggestion_run_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No enabled profiles found")

        suggestion = session.get(EntitySuggestion, suggestion_id)
        if suggestion is None:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        if suggestion.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Suggestion not found in requested profile")
        if suggestion_run_id is not None and suggestion.suggestion_run_id != suggestion_run_id:
            raise HTTPException(status_code=404, detail="Suggestion not found in requested run")

        run = session.get(EntitySuggestionRun, suggestion.suggestion_run_id)
        snapshot = session.get(EntitySnapshot, suggestion.entity_snapshot_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Suggestion run not found")

        back_query = build_query(profile_id=suggestion.profile_id, suggestion_run_id=suggestion.suggestion_run_id)
        return render_template(
            request,
            "entity_suggestion_detail.html",
            with_navigation(
                request,
                session,
                {
                    "suggestion": suggestion,
                    "run": run,
                    "snapshot": snapshot,
                    "missing_fields_pretty": safe_pretty_json(suggestion.missing_fields_json),
                    "issues_pretty": safe_pretty_json(suggestion.issues_json),
                    "semantic_type_pretty": safe_pretty_json(suggestion.semantic_type_json),
                    "llm_suggestions_pretty": safe_pretty_json(suggestion.llm_suggestions_json),
                    "source_metadata_pretty": safe_pretty_json(suggestion.source_metadata_json),
                    "back_url": f"/entity-suggestions?{back_query}",
                },
                active_profile,
            ),
        )

    @app.post("/profiles/{profile_id}/generate-automation-drafts")
    async def generate_automation_drafts(
        profile_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/automation-drafts"),
        suggestion_run_id: str = Form(default=""),
        readiness_status: str = Form(default="ready"),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        profile = session.get(Profile, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found")
        if not require_enabled_profile(profile, request):
            active_profile = choose_active_profile(session, request, None)
            return RedirectResponse(
                url=with_profile_id(next_url, active_profile.id if active_profile else None),
                status_code=303,
            )

        normalized_readiness = readiness_status.strip()
        if normalized_readiness not in READINESS_STATUSES:
            normalized_readiness = "ready"

        selected_run: EntitySuggestionRun | None = None
        normalized_run_id = suggestion_run_id.strip()
        if normalized_run_id:
            try:
                parsed_run_id = int(normalized_run_id)
            except ValueError:
                set_flash(request, "error", "Invalid suggestion run selection.")
                return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)
            candidate_run = session.get(EntitySuggestionRun, parsed_run_id)
            if candidate_run is not None and candidate_run.profile_id == profile.id:
                selected_run = candidate_run
        if selected_run is None:
            selected_run = get_latest_suggestion_run(session, profile.id)
        if selected_run is None:
            set_flash(request, "error", "No suggestion runs available. Run suggestions first.")
            return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)

        limit = draft_limit_from_env()
        candidates = list(
            session.exec(
                select(EntitySuggestion)
                .where(
                    EntitySuggestion.profile_id == profile.id,
                    EntitySuggestion.suggestion_run_id == selected_run.id,
                    EntitySuggestion.readiness_status == normalized_readiness,
                )
                .order_by(EntitySuggestion.entity_id)
                .limit(limit)
            ).all()
        )
        if not candidates:
            set_flash(
                request,
                "error",
                f"No suggestion candidates matched readiness status '{normalized_readiness}'.",
            )
            return RedirectResponse(url=with_profile_id(next_url, profile.id), status_code=303)

        pulled_at = utcnow()
        start = perf_counter()
        llm_settings = load_llm_settings()
        use_llm = llm_is_configured(llm_settings)
        llm_client = OpenAICompatibleLLMClient(llm_settings) if use_llm else None
        llm_semaphore = asyncio.Semaphore(llm_settings.max_concurrency) if use_llm else None

        draft_run = AutomationDraftRun(
            profile_id=profile.id,
            suggestion_run_id=selected_run.id,
            pulled_at=pulled_at,
            candidate_count=len(candidates),
            generated_count=0,
            error_count=0,
            duration_ms=0,
            status="success",
            error=None,
            llm_model=llm_settings.model if use_llm else None,
        )
        session.add(draft_run)
        session.commit()
        session.refresh(draft_run)
        draft_run_id = draft_run.id
        if draft_run_id is None:
            raise HTTPException(status_code=500, detail="Draft run ID is unavailable")

        snapshot_ids = [item.entity_snapshot_id for item in candidates]
        snapshots = list(
            session.exec(
                select(EntitySnapshot).where(
                    EntitySnapshot.profile_id == profile.id,
                    EntitySnapshot.sync_run_id == selected_run.sync_run_id,
                    EntitySnapshot.id.in_(snapshot_ids),
                )
            ).all()
        )
        snapshots_by_id: dict[int, EntitySnapshot] = {item.id: item for item in snapshots if item.id is not None}
        all_sync_snapshots = list(
            session.exec(
                select(EntitySnapshot).where(
                    EntitySnapshot.profile_id == profile.id,
                    EntitySnapshot.sync_run_id == selected_run.sync_run_id,
                )
            ).all()
        )

        log_event(
            "automation_draft_run_started",
            request_id=getattr(request.state, "request_id", None),
            profile_id=profile.id,
            profile_name=profile.name,
            suggestion_run_id=selected_run.id,
            draft_run_id=draft_run_id,
            candidate_count=len(candidates),
            llm_enabled=use_llm,
        )

        async def build_single_draft(suggestion_item: EntitySuggestion) -> AutomationDraft:
            snapshot = snapshots_by_id.get(suggestion_item.entity_snapshot_id)
            if snapshot is None:
                return AutomationDraft(
                    profile_id=profile.id,
                    draft_run_id=draft_run_id,
                    suggestion_run_id=selected_run.id,
                    entity_suggestion_id=suggestion_item.id if suggestion_item.id is not None else 0,
                    entity_id=suggestion_item.entity_id,
                    template_id="unknown",
                    title=f"Missing snapshot for {suggestion_item.entity_id}",
                    yaml_text=None,
                    structured_json=None,
                    rationale_json=safe_json_dump({"reason": "missing_snapshot"}),
                    generation_status="error",
                    generation_error="missing_snapshot",
                    review_status="pending",
                    pulled_at=pulled_at,
                )

            peers: list[EntitySnapshot] = []
            if snapshot.area_id:
                peers = [
                    peer
                    for peer in all_sync_snapshots
                    if peer.entity_id != snapshot.entity_id and peer.area_id == snapshot.area_id
                ]
            elif snapshot.area_name:
                peers = [
                    peer
                    for peer in all_sync_snapshots
                    if peer.entity_id != snapshot.entity_id and peer.area_name == snapshot.area_name
                ]

            return await build_automation_draft_from_suggestion(
                suggestion=suggestion_item,
                entity_snapshot=snapshot,
                peer_snapshots=peers,
                profile_id=profile.id,
                draft_run_id=draft_run_id,
                suggestion_run_id=selected_run.id,
                pulled_at=pulled_at,
                llm_client=llm_client,
                llm_semaphore=llm_semaphore,
            )

        drafts = await asyncio.gather(*[build_single_draft(item) for item in candidates])
        draft_run.generated_count = sum(1 for item in drafts if item.generation_status == "success")
        draft_run.error_count = len(drafts) - draft_run.generated_count
        draft_run.duration_ms = int((perf_counter() - start) * 1000)

        if draft_run.error_count == 0:
            draft_run.status = "success"
            draft_run.error = None
        elif draft_run.generated_count > 0:
            draft_run.status = "partial"
            draft_run.error = f"draft_errors:{draft_run.error_count}"
        else:
            draft_run.status = "failed"
            draft_run.error = "No automation drafts were generated successfully."

        session.add_all(drafts)
        session.add(draft_run)
        session.commit()

        if draft_run.status == "success":
            set_flash(request, "success", f"Generated {draft_run.generated_count} automation drafts.")
            log_event(
                "automation_draft_run_completed",
                request_id=getattr(request.state, "request_id", None),
                profile_id=profile.id,
                profile_name=profile.name,
                suggestion_run_id=selected_run.id,
                draft_run_id=draft_run_id,
                candidate_count=draft_run.candidate_count,
                generated_count=draft_run.generated_count,
                error_count=draft_run.error_count,
                duration_ms=draft_run.duration_ms,
            )
        else:
            set_flash(
                request,
                "success" if draft_run.status == "partial" else "error",
                f"Draft generation finished with status {draft_run.status}: "
                f"{draft_run.generated_count} success / {draft_run.error_count} errors.",
            )
            log_event(
                "automation_draft_run_failed",
                request_id=getattr(request.state, "request_id", None),
                profile_id=profile.id,
                profile_name=profile.name,
                suggestion_run_id=selected_run.id,
                draft_run_id=draft_run_id,
                error=draft_run.error,
                candidate_count=draft_run.candidate_count,
                generated_count=draft_run.generated_count,
                error_count=draft_run.error_count,
                duration_ms=draft_run.duration_ms,
            )

        next_target = with_profile_id(next_url, profile.id)
        split_result = urlsplit(next_target)
        query_items = [
            (k, v)
            for k, v in parse_qsl(split_result.query, keep_blank_values=True)
            if k != "draft_run_id"
        ]
        query_items.append(("draft_run_id", str(draft_run_id)))
        next_target = urlunsplit(
            ("", "", split_result.path or "/automation-drafts", urlencode(query_items), "")
        )
        return RedirectResponse(url=next_target, status_code=303)

    @app.get("/automation-drafts", response_class=HTMLResponse)
    async def list_automation_drafts(
        request: Request,
        profile_id: int | None = Query(default=None),
        draft_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        review_status: str = Query(default=""),
        template_id: str = Query(default=""),
        generation_status: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        normalized_review_status = review_status.strip()
        if normalized_review_status and normalized_review_status not in REVIEW_STATUSES:
            normalized_review_status = ""

        normalized_generation_status = generation_status.strip()
        if normalized_generation_status and normalized_generation_status not in GENERATION_STATUSES:
            normalized_generation_status = ""

        active_profile = choose_active_profile(session, request, profile_id)
        profile_count = int(session.exec(select(func.count()).select_from(Profile)).one())

        draft_runs: list[AutomationDraftRun] = []
        active_draft_run: AutomationDraftRun | None = None
        if active_profile is not None:
            draft_runs = list(
                session.exec(
                    select(AutomationDraftRun)
                    .where(AutomationDraftRun.profile_id == active_profile.id)
                    .order_by(AutomationDraftRun.pulled_at.desc(), AutomationDraftRun.id.desc())
                ).all()
            )
            if draft_run_id is not None:
                candidate_run = session.get(AutomationDraftRun, draft_run_id)
                if candidate_run is not None and candidate_run.profile_id == active_profile.id:
                    active_draft_run = candidate_run
            if active_draft_run is None:
                active_draft_run = get_latest_draft_run(session, active_profile.id)

        drafts: list[AutomationDraft] = []
        total = 0
        total_pages = 1
        review_statuses: list[str] = []
        template_ids: list[str] = []
        generation_statuses: list[str] = []
        if active_profile is not None and active_draft_run is not None:
            filtered_stmt = build_automation_draft_stmt(
                profile_id=active_profile.id,
                draft_run_id=active_draft_run.id,
                q=q,
                review_status=normalized_review_status,
                template_id=template_id,
                generation_status=normalized_generation_status,
            )
            count_stmt = select(func.count()).select_from(filtered_stmt.subquery())
            total = int(session.exec(count_stmt).one())

            total_pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, total_pages)
            offset = (page - 1) * page_size
            drafts = list(
                session.exec(
                    filtered_stmt.order_by(AutomationDraft.entity_id).offset(offset).limit(page_size)
                ).all()
            )

            review_statuses = list(
                session.exec(
                    select(AutomationDraft.review_status)
                    .where(
                        AutomationDraft.profile_id == active_profile.id,
                        AutomationDraft.draft_run_id == active_draft_run.id,
                    )
                    .distinct()
                    .order_by(AutomationDraft.review_status)
                ).all()
            )
            template_ids = list(
                session.exec(
                    select(AutomationDraft.template_id)
                    .where(
                        AutomationDraft.profile_id == active_profile.id,
                        AutomationDraft.draft_run_id == active_draft_run.id,
                    )
                    .distinct()
                    .order_by(AutomationDraft.template_id)
                ).all()
            )
            generation_statuses = list(
                session.exec(
                    select(AutomationDraft.generation_status)
                    .where(
                        AutomationDraft.profile_id == active_profile.id,
                        AutomationDraft.draft_run_id == active_draft_run.id,
                    )
                    .distinct()
                    .order_by(AutomationDraft.generation_status)
                ).all()
            )

        next_url_query = build_query(
            profile_id=active_profile.id if active_profile else None,
            draft_run_id=active_draft_run.id if active_draft_run else None,
            q=q,
            review_status=normalized_review_status,
            template_id=template_id,
            generation_status=normalized_generation_status,
            page=page,
            page_size=page_size,
        )
        next_url = f"/automation-drafts?{next_url_query}" if next_url_query else "/automation-drafts"

        return render_template(
            request,
            "automation_drafts.html",
            with_navigation(
                request,
                session,
                {
                    "active_profile": active_profile,
                    "draft_runs": draft_runs,
                    "active_draft_run": active_draft_run,
                    "drafts": drafts,
                    "q": q,
                    "review_status": normalized_review_status,
                    "template_id": template_id,
                    "generation_status": normalized_generation_status,
                    "review_statuses": review_statuses,
                    "template_ids": template_ids,
                    "generation_statuses": generation_statuses,
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages,
                    "next_url": next_url,
                    "profile_count": profile_count,
                    "profiles": get_enabled_profiles(session),
                },
                active_profile,
            ),
        )

    @app.get("/automation-drafts/{draft_id}", response_class=HTMLResponse)
    async def automation_draft_detail(
        draft_id: int,
        request: Request,
        profile_id: int | None = Query(default=None),
        draft_run_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No enabled profiles found")

        draft = session.get(AutomationDraft, draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="Draft not found")
        if draft.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Draft not found in requested profile")
        if draft_run_id is not None and draft.draft_run_id != draft_run_id:
            raise HTTPException(status_code=404, detail="Draft not found in requested run")

        run = session.get(AutomationDraftRun, draft.draft_run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Draft run not found")

        back_query = build_query(profile_id=draft.profile_id, draft_run_id=draft.draft_run_id)
        return render_template(
            request,
            "automation_draft_detail.html",
            with_navigation(
                request,
                session,
                {
                    "draft": draft,
                    "run": run,
                    "structured_pretty": safe_pretty_json(draft.structured_json),
                    "rationale_pretty": safe_pretty_json(draft.rationale_json),
                    "back_url": f"/automation-drafts?{back_query}",
                },
                active_profile,
            ),
        )

    @app.post("/automation-drafts/{draft_id}/accept")
    async def accept_automation_draft(
        draft_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/automation-drafts"),
        profile_id: str = Form(default=""),
        review_note: str = Form(default=""),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        resolved_profile_id: int | None = None
        if profile_id.strip():
            try:
                resolved_profile_id = int(profile_id.strip())
            except ValueError:
                resolved_profile_id = None
        active_profile = choose_active_profile(session, request, resolved_profile_id)
        draft = session.get(AutomationDraft, draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="Draft not found")
        if active_profile is None or draft.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Draft not found in active profile")

        draft.review_status = "accepted"
        draft.review_note = review_note.strip() or None
        draft.reviewed_at = utcnow()
        session.add(draft)
        session.commit()

        set_flash(request, "success", f"Accepted draft for {draft.entity_id}.")
        return RedirectResponse(url=with_profile_id(next_url, draft.profile_id), status_code=303)

    @app.post("/automation-drafts/{draft_id}/reject")
    async def reject_automation_draft(
        draft_id: int,
        request: Request,
        csrf_token: str = Form(...),
        next_url: str = Form(default="/automation-drafts"),
        profile_id: str = Form(default=""),
        review_note: str = Form(default=""),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        verify_csrf(request, csrf_token)

        resolved_profile_id: int | None = None
        if profile_id.strip():
            try:
                resolved_profile_id = int(profile_id.strip())
            except ValueError:
                resolved_profile_id = None
        active_profile = choose_active_profile(session, request, resolved_profile_id)
        draft = session.get(AutomationDraft, draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="Draft not found")
        if active_profile is None or draft.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Draft not found in active profile")

        draft.review_status = "rejected"
        draft.review_note = review_note.strip() or None
        draft.reviewed_at = utcnow()
        session.add(draft)
        session.commit()

        set_flash(request, "success", f"Rejected draft for {draft.entity_id}.")
        return RedirectResponse(url=with_profile_id(next_url, draft.profile_id), status_code=303)

    @app.get("/api/entity-suggestions")
    async def api_entity_suggestions(
        request: Request,
        profile_id: int | None = Query(default=None),
        suggestion_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        status: str = Query(default=""),
        domain: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> JSONResponse:
        normalized_status = status.strip()
        if normalized_status and normalized_status not in READINESS_STATUSES:
            normalized_status = ""

        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No profile found")

        active_run: EntitySuggestionRun | None = None
        if suggestion_run_id is not None:
            candidate_run = session.get(EntitySuggestionRun, suggestion_run_id)
            if candidate_run is not None and candidate_run.profile_id == active_profile.id:
                active_run = candidate_run
        if active_run is None:
            active_run = get_latest_suggestion_run(session, active_profile.id)
        if active_run is None:
            return JSONResponse(
                {
                    "profile_id": active_profile.id,
                    "profile_name": active_profile.name,
                    "suggestion_run": None,
                    "items": [],
                    "total": 0,
                }
            )

        filtered_stmt = build_entity_suggestion_stmt(
            profile_id=active_profile.id,
            suggestion_run_id=active_run.id,
            q=q,
            readiness_status=normalized_status,
            domain=domain,
        )
        total = int(session.exec(select(func.count()).select_from(filtered_stmt.subquery())).one())
        offset = (page - 1) * page_size
        rows = list(
            session.exec(filtered_stmt.order_by(EntitySuggestion.entity_id).offset(offset).limit(page_size)).all()
        )

        items: list[dict[str, Any]] = []
        for row in rows:
            items.append(
                {
                    "id": row.id,
                    "entity_id": row.entity_id,
                    "domain": row.domain,
                    "readiness_status": row.readiness_status,
                    "workflow_status": row.workflow_status,
                    "workflow_error": row.workflow_error,
                    "workflow_updated_at": (
                        row.workflow_updated_at.isoformat() if row.workflow_updated_at else None
                    ),
                    "missing_fields": safe_json_load(row.missing_fields_json, []),
                    "issues": safe_json_load(row.issues_json, []),
                    "semantic_type": safe_json_load(row.semantic_type_json, {}),
                    "llm_suggestions": safe_json_load(row.llm_suggestions_json, {}),
                    "pulled_at": row.pulled_at.isoformat(),
                }
            )

        return JSONResponse(
            {
                "profile_id": active_profile.id,
                "profile_name": active_profile.name,
                "suggestion_run": {
                    "id": active_run.id,
                    "sync_run_id": active_run.sync_run_id,
                    "status": active_run.status,
                    "entity_count": active_run.entity_count,
                    "ready_count": active_run.ready_count,
                    "needs_review_count": active_run.needs_review_count,
                    "blocked_count": active_run.blocked_count,
                    "policy_version": active_run.policy_version,
                    "llm_enabled": active_run.llm_enabled,
                    "llm_model": active_run.llm_model,
                    "pulled_at": active_run.pulled_at.isoformat(),
                },
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": items,
            }
        )

    @app.get("/api/entity-suggestions/{suggestion_id}")
    async def api_entity_suggestion_detail(
        suggestion_id: int,
        request: Request,
        profile_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> JSONResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No profile found")

        suggestion = session.get(EntitySuggestion, suggestion_id)
        if suggestion is None or suggestion.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        return JSONResponse(
            {
                "id": suggestion.id,
                "profile_id": suggestion.profile_id,
                "sync_run_id": suggestion.sync_run_id,
                "suggestion_run_id": suggestion.suggestion_run_id,
                "entity_snapshot_id": suggestion.entity_snapshot_id,
                "entity_id": suggestion.entity_id,
                "domain": suggestion.domain,
                "readiness_status": suggestion.readiness_status,
                "workflow_status": suggestion.workflow_status,
                "workflow_error": suggestion.workflow_error,
                "workflow_updated_at": (
                    suggestion.workflow_updated_at.isoformat() if suggestion.workflow_updated_at else None
                ),
                "missing_fields": safe_json_load(suggestion.missing_fields_json, []),
                "issues": safe_json_load(suggestion.issues_json, []),
                "semantic_type": safe_json_load(suggestion.semantic_type_json, {}),
                "llm_suggestions": safe_json_load(suggestion.llm_suggestions_json, {}),
                "source_metadata": safe_json_load(suggestion.source_metadata_json, {}),
                "pulled_at": suggestion.pulled_at.isoformat(),
            }
        )

    @app.get("/api/automation-drafts")
    async def api_automation_drafts(
        request: Request,
        profile_id: int | None = Query(default=None),
        draft_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        review_status: str = Query(default=""),
        template_id: str = Query(default=""),
        generation_status: str = Query(default=""),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> JSONResponse:
        normalized_review_status = review_status.strip()
        if normalized_review_status and normalized_review_status not in REVIEW_STATUSES:
            normalized_review_status = ""
        normalized_generation_status = generation_status.strip()
        if normalized_generation_status and normalized_generation_status not in GENERATION_STATUSES:
            normalized_generation_status = ""

        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No profile found")

        active_run: AutomationDraftRun | None = None
        if draft_run_id is not None:
            candidate_run = session.get(AutomationDraftRun, draft_run_id)
            if candidate_run is not None and candidate_run.profile_id == active_profile.id:
                active_run = candidate_run
        if active_run is None:
            active_run = get_latest_draft_run(session, active_profile.id)
        if active_run is None:
            return JSONResponse(
                {
                    "profile_id": active_profile.id,
                    "profile_name": active_profile.name,
                    "draft_run": None,
                    "items": [],
                    "total": 0,
                }
            )

        filtered_stmt = build_automation_draft_stmt(
            profile_id=active_profile.id,
            draft_run_id=active_run.id,
            q=q,
            review_status=normalized_review_status,
            template_id=template_id,
            generation_status=normalized_generation_status,
        )
        total = int(session.exec(select(func.count()).select_from(filtered_stmt.subquery())).one())
        offset = (page - 1) * page_size
        rows = list(
            session.exec(filtered_stmt.order_by(AutomationDraft.entity_id).offset(offset).limit(page_size)).all()
        )

        items: list[dict[str, Any]] = []
        for row in rows:
            items.append(
                {
                    "id": row.id,
                    "entity_id": row.entity_id,
                    "template_id": row.template_id,
                    "title": row.title,
                    "generation_status": row.generation_status,
                    "generation_error": row.generation_error,
                    "review_status": row.review_status,
                    "review_note": row.review_note,
                    "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
                    "pulled_at": row.pulled_at.isoformat(),
                }
            )

        return JSONResponse(
            {
                "profile_id": active_profile.id,
                "profile_name": active_profile.name,
                "draft_run": {
                    "id": active_run.id,
                    "suggestion_run_id": active_run.suggestion_run_id,
                    "status": active_run.status,
                    "candidate_count": active_run.candidate_count,
                    "generated_count": active_run.generated_count,
                    "error_count": active_run.error_count,
                    "llm_model": active_run.llm_model,
                    "pulled_at": active_run.pulled_at.isoformat(),
                },
                "total": total,
                "page": page,
                "page_size": page_size,
                "items": items,
            }
        )

    @app.get("/api/automation-drafts/{draft_id}")
    async def api_automation_draft_detail(
        draft_id: int,
        request: Request,
        profile_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> JSONResponse:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No profile found")

        draft = session.get(AutomationDraft, draft_id)
        if draft is None or draft.profile_id != active_profile.id:
            raise HTTPException(status_code=404, detail="Draft not found")

        return JSONResponse(
            {
                "id": draft.id,
                "profile_id": draft.profile_id,
                "draft_run_id": draft.draft_run_id,
                "suggestion_run_id": draft.suggestion_run_id,
                "entity_suggestion_id": draft.entity_suggestion_id,
                "entity_id": draft.entity_id,
                "template_id": draft.template_id,
                "title": draft.title,
                "yaml_text": draft.yaml_text,
                "structured": safe_json_load(draft.structured_json, {}),
                "rationale": safe_json_load(draft.rationale_json, {}),
                "generation_status": draft.generation_status,
                "generation_error": draft.generation_error,
                "review_status": draft.review_status,
                "review_note": draft.review_note,
                "reviewed_at": draft.reviewed_at.isoformat() if draft.reviewed_at else None,
                "pulled_at": draft.pulled_at.isoformat(),
            }
        )

    def get_export_rows(
        session: Session,
        active_profile: Profile,
        active_sync_run: SyncRun,
        q: str,
        domain: str,
        state_value: str,
        changed_within: int | None,
    ) -> list[EntitySnapshot]:
        stmt = build_entity_stmt(
            profile_id=active_profile.id,
            sync_run_id=active_sync_run.id,
            q=q,
            domain=domain,
            state_value=state_value,
            changed_within=changed_within,
        )
        return session.exec(stmt.order_by(EntitySnapshot.entity_id)).all()

    @app.get("/export/json")
    async def export_json(
        request: Request,
        profile_id: int | None = Query(default=None),
        sync_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        domain: str = Query(default=""),
        state_value: str = Query(default="", alias="state"),
        changed_within: int | None = Query(default=None, ge=1, le=10080),
        session: Session = Depends(get_session),
    ) -> Response:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No profile found")

        active_sync_run: SyncRun | None = None
        if sync_run_id is not None:
            candidate = session.get(SyncRun, sync_run_id)
            if candidate is not None and candidate.profile_id == active_profile.id:
                active_sync_run = candidate
        if active_sync_run is None:
            active_sync_run = get_latest_sync_run(session, active_profile.id)
        if active_sync_run is None:
            raise HTTPException(status_code=404, detail="No synced entities found")

        rows = get_export_rows(
            session,
            active_profile,
            active_sync_run,
            q=q,
            domain=domain,
            state_value=state_value,
            changed_within=changed_within,
        )

        payload: list[dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "profile_id": row.profile_id,
                    "profile_name": active_profile.name,
                    "sync_run_id": row.sync_run_id,
                    "pulled_at": row.pulled_at.isoformat(),
                    "entity_id": row.entity_id,
                    "domain": row.domain,
                    "state": row.state,
                    "friendly_name": row.friendly_name,
                    "device_id": row.device_id,
                    "device_name": row.device_name,
                    "area_id": row.area_id,
                    "area_name": row.area_name,
                    "floor_id": row.floor_id,
                    "floor_name": row.floor_name,
                    "location_name": row.location_name,
                    "labels": json.loads(row.labels_json) if row.labels_json else None,
                    "metadata": json.loads(row.metadata_json) if row.metadata_json else None,
                    "attributes": json.loads(row.attributes_json),
                    "context": json.loads(row.context_json) if row.context_json else None,
                    "last_changed": row.last_changed.isoformat() if row.last_changed else None,
                    "last_updated": row.last_updated.isoformat() if row.last_updated else None,
                }
            )

        filename = (
            f"entities_{active_profile.name}_{active_sync_run.pulled_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        return Response(
            content=json.dumps(payload, indent=2, ensure_ascii=True),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/export/csv")
    async def export_csv(
        request: Request,
        profile_id: int | None = Query(default=None),
        sync_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        domain: str = Query(default=""),
        state_value: str = Query(default="", alias="state"),
        changed_within: int | None = Query(default=None, ge=1, le=10080),
        session: Session = Depends(get_session),
    ) -> Response:
        active_profile = choose_active_profile(session, request, profile_id)
        if active_profile is None:
            raise HTTPException(status_code=404, detail="No profile found")

        active_sync_run: SyncRun | None = None
        if sync_run_id is not None:
            candidate = session.get(SyncRun, sync_run_id)
            if candidate is not None and candidate.profile_id == active_profile.id:
                active_sync_run = candidate
        if active_sync_run is None:
            active_sync_run = get_latest_sync_run(session, active_profile.id)
        if active_sync_run is None:
            raise HTTPException(status_code=404, detail="No synced entities found")

        rows = get_export_rows(
            session,
            active_profile,
            active_sync_run,
            q=q,
            domain=domain,
            state_value=state_value,
            changed_within=changed_within,
        )

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "profile_id",
                "profile_name",
                "sync_run_id",
                "pulled_at",
                "entity_id",
                "domain",
                "state",
                "friendly_name",
                "device_id",
                "device_name",
                "area_id",
                "area_name",
                "floor_id",
                "floor_name",
                "location_name",
                "last_changed",
                "last_updated",
                "labels_json",
                "metadata_json",
                "attributes_json",
                "context_json",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "profile_id": row.profile_id,
                    "profile_name": active_profile.name,
                    "sync_run_id": row.sync_run_id,
                    "pulled_at": row.pulled_at.isoformat(),
                    "entity_id": row.entity_id,
                    "domain": row.domain,
                    "state": row.state,
                    "friendly_name": row.friendly_name or "",
                    "device_id": row.device_id or "",
                    "device_name": row.device_name or "",
                    "area_id": row.area_id or "",
                    "area_name": row.area_name or "",
                    "floor_id": row.floor_id or "",
                    "floor_name": row.floor_name or "",
                    "location_name": row.location_name or "",
                    "last_changed": row.last_changed.isoformat() if row.last_changed else "",
                    "last_updated": row.last_updated.isoformat() if row.last_updated else "",
                    "labels_json": row.labels_json or "",
                    "metadata_json": row.metadata_json or "",
                    "attributes_json": row.attributes_json,
                    "context_json": row.context_json or "",
                }
            )

        filename = (
            f"entities_{active_profile.name}_{active_sync_run.pulled_at.strftime('%Y%m%dT%H%M%SZ')}.csv"
        )
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return app


app = create_app()
