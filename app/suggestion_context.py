from __future__ import annotations

import json
from collections import Counter
from typing import Any

from sqlmodel import Session, select

from app.models import ConfigSnapshot, ConfigSyncRun, EntitySnapshot, Profile, SyncRun

MAX_CONTEXT_BYTES = 45_000
MAX_LINKED_ENTITIES = 40
MAX_RELATED_CONFIG = 30


def parse_json_or_default(raw_json: str | None, default: Any) -> Any:
    if not raw_json:
        return default
    try:
        return json.loads(raw_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def collect_reference_entity_ids(snapshot: ConfigSnapshot) -> set[str]:
    references = parse_json_or_default(snapshot.references_json, {})
    entity_ids = references.get("entity_id") if isinstance(references, dict) else None
    if not isinstance(entity_ids, list):
        return set()
    values: set[str] = set()
    for value in entity_ids:
        if isinstance(value, str) and value.strip():
            values.add(value.strip().lower())
    return values


def get_latest_entity_sync_run(session: Session, profile_id: int) -> SyncRun | None:
    stmt = (
        select(SyncRun)
        .where(SyncRun.profile_id == profile_id)
        .where(SyncRun.status == "success")
        .order_by(SyncRun.pulled_at.desc(), SyncRun.id.desc())
    )
    return session.exec(stmt).first()


def get_config_sync_run(session: Session, target: ConfigSnapshot) -> ConfigSyncRun | None:
    return session.get(ConfigSyncRun, target.config_sync_run_id)


def build_known_entity_ids(
    session: Session,
    profile_id: int,
    config_sync_run_id: int | None,
) -> set[str]:
    known_ids: set[str] = set()

    latest_sync_run = get_latest_entity_sync_run(session, profile_id)
    if latest_sync_run is not None:
        entity_ids = session.exec(
            select(EntitySnapshot.entity_id).where(
                EntitySnapshot.profile_id == profile_id,
                EntitySnapshot.sync_run_id == latest_sync_run.id,
            )
        ).all()
        for entity_id in entity_ids:
            cleaned = str(entity_id).strip().lower()
            if cleaned:
                known_ids.add(cleaned)

    if config_sync_run_id is not None:
        config_ids = session.exec(
            select(ConfigSnapshot.entity_id).where(
                ConfigSnapshot.profile_id == profile_id,
                ConfigSnapshot.config_sync_run_id == config_sync_run_id,
            )
        ).all()
        for entity_id in config_ids:
            cleaned = str(entity_id).strip().lower()
            if cleaned:
                known_ids.add(cleaned)

    return known_ids


def _context_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8"))


def _trim_context_payload(payload: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    while _context_size(payload) > max_bytes:
        linked_entities = payload.get("linked_entities")
        related_config = payload.get("related_config")
        if isinstance(linked_entities, list) and linked_entities:
            linked_entities.pop()
            continue
        if isinstance(related_config, list) and related_config:
            related_config.pop()
            continue
        target = payload.get("target")
        if isinstance(target, dict) and "config" in target and isinstance(target.get("config"), dict):
            target["config"] = {"_truncated": True}
            continue
        break
    return payload


def build_automation_context(
    session: Session,
    profile: Profile,
    target: ConfigSnapshot,
    max_context_bytes: int = MAX_CONTEXT_BYTES,
) -> dict[str, Any]:
    target_references = collect_reference_entity_ids(target)
    config_sync_run = get_config_sync_run(session, target)
    latest_entity_sync_run = get_latest_entity_sync_run(session, profile.id)

    linked_entities: list[dict[str, Any]] = []
    if latest_entity_sync_run is not None and target_references:
        entity_rows = session.exec(
            select(EntitySnapshot)
            .where(
                EntitySnapshot.profile_id == profile.id,
                EntitySnapshot.sync_run_id == latest_entity_sync_run.id,
                EntitySnapshot.entity_id.in_(sorted(target_references)),
            )
            .order_by(EntitySnapshot.entity_id)
            .limit(MAX_LINKED_ENTITIES)
        ).all()
        linked_entities = [
            {
                "entity_id": row.entity_id,
                "domain": row.domain,
                "state": row.state,
                "friendly_name": row.friendly_name,
                "area_name": row.area_name,
                "device_name": row.device_name,
                "last_updated": row.last_updated.isoformat() if row.last_updated else None,
            }
            for row in entity_rows
        ]

    related_config: list[dict[str, Any]] = []
    if config_sync_run is not None and target_references:
        config_rows = session.exec(
            select(ConfigSnapshot)
            .where(
                ConfigSnapshot.profile_id == profile.id,
                ConfigSnapshot.config_sync_run_id == config_sync_run.id,
                ConfigSnapshot.fetch_status == "success",
            )
            .order_by(ConfigSnapshot.kind, ConfigSnapshot.entity_id)
            .limit(300)
        ).all()
        for row in config_rows:
            if row.id == target.id:
                continue
            row_refs = collect_reference_entity_ids(row)
            if not row_refs.intersection(target_references):
                continue
            related_config.append(
                {
                    "entity_id": row.entity_id,
                    "kind": row.kind,
                    "name": row.name,
                    "summary": parse_json_or_default(row.summary_json, {}),
                    "references": sorted(row_refs),
                }
            )
            if len(related_config) >= MAX_RELATED_CONFIG:
                break

    domain_counts: dict[str, int] = {}
    if latest_entity_sync_run is not None:
        domains = session.exec(
            select(EntitySnapshot.domain).where(
                EntitySnapshot.profile_id == profile.id,
                EntitySnapshot.sync_run_id == latest_entity_sync_run.id,
            )
        ).all()
        counts = Counter(str(item).strip() for item in domains if str(item).strip())
        domain_counts = dict(counts.most_common(20))

    payload: dict[str, Any] = {
        "schema_version": "haev.automation.context.v1",
        "profile": {
            "id": profile.id,
            "name": profile.name,
        },
        "target": {
            "config_snapshot_id": target.id,
            "entity_id": target.entity_id,
            "kind": target.kind,
            "name": target.name,
            "summary": parse_json_or_default(target.summary_json, {}),
            "references": sorted(target_references),
            "config": parse_json_or_default(target.config_json, {}),
            "attributes": parse_json_or_default(target.attributes_json, {}),
        },
        "linked_entities": linked_entities,
        "related_config": related_config,
        "profile_summary": {
            "latest_entity_sync_run_id": latest_entity_sync_run.id if latest_entity_sync_run else None,
            "latest_config_sync_run_id": config_sync_run.id if config_sync_run else None,
            "domain_counts": domain_counts,
            "linked_entity_count": len(linked_entities),
            "related_config_count": len(related_config),
        },
    }
    return _trim_context_payload(payload, max_context_bytes)
