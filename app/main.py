from __future__ import annotations

import csv
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
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import delete, func, or_
from sqlmodel import Session, select
from starlette.middleware.sessions import SessionMiddleware

from app.db import ensure_default_profile, get_session, run_migrations
from app.ha_client import HAClient, HAClientError
from app.models import EntitySnapshot, Profile, SyncRun, utcnow

logger = logging.getLogger("ha_entity_vault")
CSRF_SESSION_KEY = "csrf_token"
FLASH_SESSION_KEY = "flash"
DEFAULT_PAGE_SIZE = 50


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


def choose_active_profile(session: Session, profile_id: int | None) -> Profile | None:
    if profile_id is not None:
        return session.get(Profile, profile_id)

    default_profile = session.exec(select(Profile).where(Profile.name == "default")).first()
    if default_profile is not None:
        return default_profile

    return session.exec(select(Profile).order_by(Profile.id)).first()


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


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    run_migrations()
    ensure_default_profile()
    yield


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
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        profiles = session.exec(select(Profile).order_by(Profile.name)).all()
        return render_template(
            request,
            "settings.html",
            {
                "profiles": profiles,
            },
        )

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
        session.exec(delete(EntitySnapshot).where(EntitySnapshot.profile_id == profile_id))
        session.exec(delete(SyncRun).where(SyncRun.profile_id == profile_id))
        session.delete(profile)
        session.commit()

        remaining_profiles = session.exec(select(func.count()).select_from(Profile)).one()
        if int(remaining_profiles) == 0:
            default_profile = Profile(
                name="default",
                base_url="http://homeassistant.local:8123",
                token="",
                token_env_var="HA_TOKEN",
                verify_tls=True,
                timeout_seconds=10,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
            session.add(default_profile)
            session.commit()

        set_flash(request, "success", f"Profile '{deleted_name}' deleted.")
        return RedirectResponse(url="/settings", status_code=303)

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

        token = resolve_profile_token(profile)
        if not token:
            set_flash(request, "error", "No token configured in profile or environment.")
            return RedirectResponse(url=normalize_next_url(next_url), status_code=303)

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

        return RedirectResponse(url=normalize_next_url(next_url), status_code=303)

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

        token = resolve_profile_token(profile)
        if not token:
            set_flash(request, "error", "No token configured in profile or environment.")
            return RedirectResponse(url=normalize_next_url(next_url), status_code=303)

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
                profile_id=profile.id,
                pulled_at=pulled_at,
                entity_count=0,
                duration_ms=int((perf_counter() - start) * 1000),
                status="failed",
                error=str(exc),
            )
            session.add(failed_run)
            session.commit()

            set_flash(request, "error", f"Sync failed: {exc}")
            log_event(
                "sync_failed",
                request_id=getattr(request.state, "request_id", None),
                profile_id=profile.id,
                profile_name=profile.name,
                error=str(exc),
            )
            return RedirectResponse(url=normalize_next_url(next_url), status_code=303)

        registry_metadata: dict[str, list[dict[str, Any]]] = {}
        try:
            registry_metadata = await client.fetch_registry_metadata()
        except HAClientError as exc:
            log_event(
                "sync_registry_enrichment_unavailable",
                request_id=getattr(request.state, "request_id", None),
                profile_id=profile.id,
                profile_name=profile.name,
                error=str(exc),
            )

        registry_lookup = build_registry_lookup(registry_metadata)

        sync_run = SyncRun(
            profile_id=profile.id,
            pulled_at=pulled_at,
            entity_count=len(states),
            duration_ms=0,
            status="success",
            error=None,
        )
        session.add(sync_run)
        session.commit()
        session.refresh(sync_run)

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
                    profile_id=profile.id,
                    sync_run_id=sync_run.id,
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
            request_id=getattr(request.state, "request_id", None),
            profile_id=profile.id,
            profile_name=profile.name,
            sync_run_id=sync_run.id,
            entity_count=len(snapshots),
            duration_ms=sync_run.duration_ms,
            pulled_at=pulled_at.isoformat(),
        )
        set_flash(request, "success", f"Sync complete. Stored {len(snapshots)} entities.")

        next_target = normalize_next_url(next_url)
        if next_target.startswith("/entities") and "profile_id=" not in next_target:
            query = build_query(profile_id=profile.id)
            next_target = f"/entities?{query}" if query else "/entities"

        return RedirectResponse(url=next_target, status_code=303)

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
        profiles = session.exec(select(Profile).order_by(Profile.name)).all()
        active_profile = choose_active_profile(session, profile_id)

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
            {
                "profiles": profiles,
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
            },
        )

    @app.get("/entities/{entity_id}", response_class=HTMLResponse)
    async def entity_detail(
        entity_id: str,
        request: Request,
        profile_id: int | None = Query(default=None),
        sync_run_id: int | None = Query(default=None),
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        active_profile = choose_active_profile(session, profile_id)
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
        profile_id: int | None = Query(default=None),
        sync_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        domain: str = Query(default=""),
        state_value: str = Query(default="", alias="state"),
        changed_within: int | None = Query(default=None, ge=1, le=10080),
        session: Session = Depends(get_session),
    ) -> Response:
        active_profile = choose_active_profile(session, profile_id)
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
        profile_id: int | None = Query(default=None),
        sync_run_id: int | None = Query(default=None),
        q: str = Query(default=""),
        domain: str = Query(default=""),
        state_value: str = Query(default="", alias="state"),
        changed_within: int | None = Query(default=None, ge=1, le=10080),
        session: Session = Depends(get_session),
    ) -> Response:
        active_profile = choose_active_profile(session, profile_id)
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
