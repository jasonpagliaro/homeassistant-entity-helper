from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session

from app import db
from app.main import create_app
from app.models import ConfigSnapshot, ConfigSyncRun, EntitySnapshot, Profile, SyncRun, utcnow
from app.suggestion_context import build_automation_context, build_known_entity_ids


def test_build_automation_context_includes_linked_entities_and_related_config(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("APP_NAME", "Context Test")
    db.reset_engine_for_tests()

    app = create_app()
    with TestClient(app):
        pass

    with Session(db.get_engine()) as session:
        profile = Profile(
            name="ctx",
            base_url="http://ha.local:8123",
            token="token",
            token_env_var="HA_TOKEN",
            verify_tls=True,
            timeout_seconds=10,
            is_enabled=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(profile)
        session.commit()
        session.refresh(profile)
        assert profile.id is not None

        sync_run = SyncRun(
            profile_id=profile.id,
            pulled_at=utcnow(),
            entity_count=2,
            duration_ms=1,
            status="success",
            error=None,
        )
        session.add(sync_run)
        session.commit()
        session.refresh(sync_run)
        assert sync_run.id is not None

        session.add(
            EntitySnapshot(
                profile_id=profile.id,
                sync_run_id=sync_run.id,
                entity_id="person.jason",
                domain="person",
                state="home",
                friendly_name="Jason",
                attributes_json=json.dumps({}),
                pulled_at=utcnow(),
            )
        )
        session.add(
            EntitySnapshot(
                profile_id=profile.id,
                sync_run_id=sync_run.id,
                entity_id="light.kitchen",
                domain="light",
                state="on",
                friendly_name="Kitchen Light",
                attributes_json=json.dumps({}),
                pulled_at=utcnow(),
            )
        )
        session.commit()

        config_run = ConfigSyncRun(
            profile_id=profile.id,
            pulled_at=utcnow(),
            item_count=2,
            success_count=2,
            error_count=0,
            duration_ms=1,
            status="success",
            error=None,
        )
        session.add(config_run)
        session.commit()
        session.refresh(config_run)
        assert config_run.id is not None

        target = ConfigSnapshot(
            profile_id=profile.id,
            config_sync_run_id=config_run.id,
            kind="automation",
            entity_id="automation.evening_mode",
            config_key="evening_mode",
            name="Evening Mode",
            state="on",
            fetch_status="success",
            summary_json=json.dumps({"name": "Evening Mode"}),
            references_json=json.dumps({"entity_id": ["person.jason", "light.kitchen"]}),
            config_json=json.dumps({"alias": "Evening Mode"}),
            attributes_json=json.dumps({}),
            metadata_json=json.dumps({}),
            pulled_at=utcnow(),
        )
        related = ConfigSnapshot(
            profile_id=profile.id,
            config_sync_run_id=config_run.id,
            kind="script",
            entity_id="script.goodnight",
            config_key="goodnight",
            name="Goodnight",
            state="off",
            fetch_status="success",
            summary_json=json.dumps({"name": "Goodnight"}),
            references_json=json.dumps({"entity_id": ["light.kitchen"]}),
            config_json=json.dumps({"alias": "Goodnight"}),
            attributes_json=json.dumps({}),
            metadata_json=json.dumps({}),
            pulled_at=utcnow(),
        )
        session.add(target)
        session.add(related)
        session.commit()
        session.refresh(target)
        assert target.id is not None

        context = build_automation_context(session, profile, target, max_context_bytes=12_000)
        assert context["target"]["entity_id"] == "automation.evening_mode"
        assert any(item["entity_id"] == "person.jason" for item in context["linked_entities"])
        assert any(item["entity_id"] == "script.goodnight" for item in context["related_config"])

        known_ids = build_known_entity_ids(session, profile.id, config_run.id)
        assert "person.jason" in known_ids
        assert "automation.evening_mode" in known_ids

    db.reset_engine_for_tests()
