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
from sqlalchemy import delete, func
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
        stmt = stmt.where(func.lower(EntitySnapshot.entity_id).like(pattern))

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

            snapshots.append(
                EntitySnapshot(
                    profile_id=profile.id,
                    sync_run_id=sync_run.id,
                    entity_id=entity_id,
                    domain=domain,
                    state=str(state.get("state", "")),
                    attributes_json=safe_json_dump(state.get("attributes", {})),
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
                "last_changed",
                "last_updated",
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
                    "last_changed": row.last_changed.isoformat() if row.last_changed else "",
                    "last_updated": row.last_updated.isoformat() if row.last_updated else "",
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
