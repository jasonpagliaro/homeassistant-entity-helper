from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app import db
from app.ha_client import HAClientError
from app.llm_client import LLMClientError
from app.main import create_app
from app.models import (
    ConfigSnapshot,
    ConfigSyncRun,
    EntitySuggestion,
    LLMConnection,
    Profile,
    SuggestionProposal,
    SuggestionRun,
    utcnow,
)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("APP_NAME", "HA Entity Vault Test")

    async def default_fetch_entity_registry_entries(_: Any) -> list[dict[str, Any]]:
        return []

    async def default_fetch_device_registry_entries(_: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(
        "app.main.HAClient.fetch_entity_registry_entries",
        default_fetch_entity_registry_entries,
    )
    monkeypatch.setattr(
        "app.main.HAClient.fetch_device_registry_entries",
        default_fetch_device_registry_entries,
    )

    db.reset_engine_for_tests()
    app = create_app()

    with TestClient(app) as test_client:
        yield test_client

    db.reset_engine_for_tests()


def extract_csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def get_profile_id_by_name(name: str) -> int:
    with Session(db.get_engine()) as session:
        profile = session.exec(select(Profile).where(Profile.name == name)).first()
        assert profile is not None
        assert profile.id is not None
        return profile.id


def create_profile(
    client: TestClient,
    csrf_token: str,
    *,
    name: str,
    base_url: str = "http://ha.local:8123",
    token: str = "test-token",
) -> int:
    create_response = client.post(
        "/profiles",
        data={
            "csrf_token": csrf_token,
            "name": name,
            "base_url": base_url,
            "token": token,
            "token_env_var": "HA_TOKEN",
            "verify_tls": "on",
            "timeout_seconds": "10",
        },
        follow_redirects=False,
    )
    assert create_response.status_code == 303
    return get_profile_id_by_name(name)


def get_llm_connection_id(profile_id: int, name: str) -> int:
    with Session(db.get_engine()) as session:
        connection = session.exec(
            select(LLMConnection).where(
                LLMConnection.profile_id == profile_id,
                LLMConnection.name == name,
            )
        ).first()
        assert connection is not None
        assert connection.id is not None
        return connection.id


def seed_default_llm_connection(profile_id: int, *, name: str = "Default LLM") -> int:
    now = utcnow()
    with Session(db.get_engine()) as session:
        connection = LLMConnection(
            profile_id=profile_id,
            name=name,
            provider_kind="openai_compatible",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key_env_var="TEST_LLM_KEY",
            timeout_seconds=20,
            temperature=0.2,
            max_output_tokens=900,
            extra_headers_json=None,
            is_enabled=True,
            is_default=True,
            created_at=now,
            updated_at=now,
        )
        session.add(connection)
        session.commit()
        session.refresh(connection)
        assert connection.id is not None
        return connection.id


def seed_config_snapshot(
    profile_id: int,
    *,
    entity_id: str = "automation.evening_mode",
    kind: str = "automation",
    fetch_status: str = "success",
) -> tuple[int, int]:
    with Session(db.get_engine()) as session:
        config_run = ConfigSyncRun(
            profile_id=profile_id,
            item_count=1,
            success_count=1 if fetch_status == "success" else 0,
            error_count=0 if fetch_status == "success" else 1,
            duration_ms=1,
            status="success",
            error=None,
            pulled_at=utcnow(),
        )
        session.add(config_run)
        session.commit()
        session.refresh(config_run)
        assert config_run.id is not None

        snapshot = ConfigSnapshot(
            profile_id=profile_id,
            config_sync_run_id=config_run.id,
            kind=kind,
            entity_id=entity_id,
            config_key="auto_evening_mode",
            name="Evening Mode",
            state="on",
            fetch_status=fetch_status,
            fetch_error=None if fetch_status == "success" else "failed",
            summary_json=json.dumps({"name": "Evening Mode"}),
            references_json=json.dumps({"entity_id": ["person.jason"]}),
            config_json=json.dumps(
                {
                    "alias": "Evening Mode",
                    "trigger": [{"platform": "time", "at": "19:00:00"}],
                    "condition": [],
                    "action": [{"service": "light.turn_on", "target": {"entity_id": "light.kitchen"}}],
                }
            ),
            attributes_json=json.dumps({"friendly_name": "Evening Mode"}),
            metadata_json=json.dumps({"detail_source": "test"}),
            pulled_at=config_run.pulled_at,
        )
        session.add(snapshot)
        session.commit()
        session.refresh(snapshot)
        assert snapshot.id is not None
        return config_run.id, snapshot.id


def seed_suggestion_run_with_proposals(
    profile_id: int,
    connection_id: int,
    *,
    run_status: str = "succeeded",
) -> tuple[int, list[int]]:
    now = utcnow()
    with Session(db.get_engine()) as session:
        run = SuggestionRun(
            profile_id=profile_id,
            llm_connection_id=connection_id,
            config_sync_run_id=None,
            run_kind="concept_v2",
            idea_type="general",
            custom_intent=None,
            mode="standard",
            top_k=10,
            include_existing=True,
            include_new=True,
            status=run_status,
            target_count=3,
            processed_count=3 if run_status in {"succeeded", "failed"} else 1,
            success_count=1,
            invalid_count=1,
            error_count=1 if run_status == "failed" else 0,
            error="synthetic_failure" if run_status == "failed" else None,
            context_hash=None,
            filters_json=json.dumps({"idea_type": "general"}),
            result_summary_json=json.dumps({"target_count": 3}),
            started_at=now,
            finished_at=now if run_status in {"succeeded", "failed"} else None,
            created_at=now,
            updated_at=now,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        assert run.id is not None

        proposals = [
            SuggestionProposal(
                profile_id=profile_id,
                suggestion_run_id=run.id,
                config_snapshot_id=None,
                target_entity_id="automation.kitchen_alert",
                status="proposed",
                summary="Kitchen alert concept.",
                confidence=0.82,
                risk_level="low",
                concept_payload_json=json.dumps(
                    {
                        "title": "Kitchen Alert",
                        "concept_type": "safety",
                        "target_kind": "new_automation",
                        "target_entity_id": "automation.kitchen_alert",
                    }
                ),
                concept_type="safety",
                impact_score=0.81,
                feasibility_score=0.79,
                novelty_score=0.52,
                ranking_score=0.76,
                ranking_breakdown_json=json.dumps({"weighted_total": 0.76}),
                duplicate_fingerprint=None,
                queue_stage="suggested",
                queue_note=None,
                queue_updated_at=now,
                proposed_patch_json=None,
                verification_steps_json=json.dumps(["Simulate trigger in HA UI"]),
                raw_response_json=json.dumps({}),
                validation_error=None,
                created_at=now,
                updated_at=now,
            ),
            SuggestionProposal(
                profile_id=profile_id,
                suggestion_run_id=run.id,
                config_snapshot_id=None,
                target_entity_id="automation.night_lock_check",
                status="accepted",
                summary="Night lock reminder",
                confidence=0.74,
                risk_level="low",
                concept_payload_json=json.dumps(
                    {
                        "title": "Night Lock Check",
                        "concept_type": "security",
                        "target_kind": "new_automation",
                        "target_entity_id": "automation.night_lock_check",
                    }
                ),
                concept_type="security",
                impact_score=0.69,
                feasibility_score=0.85,
                novelty_score=0.44,
                ranking_score=0.71,
                ranking_breakdown_json=json.dumps({"weighted_total": 0.71}),
                duplicate_fingerprint=None,
                queue_stage="queued",
                queue_note=None,
                queue_updated_at=now,
                proposed_patch_json=None,
                verification_steps_json=json.dumps(["Run lock status simulation"]),
                raw_response_json=json.dumps({}),
                validation_error=None,
                created_at=now,
                updated_at=now,
            ),
            SuggestionProposal(
                profile_id=profile_id,
                suggestion_run_id=run.id,
                config_snapshot_id=None,
                target_entity_id="automation.invalid_response",
                status="invalid",
                summary="Invalid response placeholder",
                confidence=None,
                risk_level=None,
                concept_payload_json=None,
                concept_type=None,
                impact_score=None,
                feasibility_score=None,
                novelty_score=None,
                ranking_score=None,
                ranking_breakdown_json=None,
                duplicate_fingerprint=None,
                queue_stage="archived",
                queue_note=None,
                queue_updated_at=now,
                proposed_patch_json=None,
                verification_steps_json=None,
                raw_response_json=json.dumps({"reason": "schema_invalid"}),
                validation_error="response_missing_suggestions_array",
                created_at=now,
                updated_at=now,
            ),
        ]
        session.add_all(proposals)
        session.commit()
        for item in proposals:
            session.refresh(item)
            assert item.id is not None

        return run.id, [item.id for item in proposals if item.id is not None]


def wait_for_suggestion_run(
    client: TestClient,
    profile_id: int,
    run_id: int,
    *,
    attempts: int = 40,
    delay_seconds: float = 0.05,
) -> dict[str, Any]:
    for _ in range(attempts):
        response = client.get(f"/api/suggestions/runs/{run_id}?profile_id={profile_id}")
        assert response.status_code == 200
        payload = response.json()
        status = payload["run"]["status"]
        if status not in {"queued", "running"}:
            return payload
        time.sleep(delay_seconds)
    pytest.fail(f"Suggestion run {run_id} did not complete in time")


def run_sync_and_suggestions(client: TestClient, profile_id: int, csrf_token: str) -> None:
    sync_response = client.post(
        f"/profiles/{profile_id}/sync",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entities?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert sync_response.status_code == 303

    suggestion_response = client.post(
        f"/profiles/{profile_id}/run-entity-suggestions",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entity-suggestions?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert suggestion_response.status_code == 303


def assert_sync_modal_markup(html: str) -> None:
    assert 'id="sync-modal"' in html
    assert 'id="sync-modal-label"' in html
    assert "const DEFAULT_DELAY_MS = 300;" in html


def assert_form_has_sync_modal_attrs(html: str, action_pattern: str, label: str) -> None:
    match = re.search(rf'<form[^>]*action="{action_pattern}"[^>]*>', html)
    assert match is not None
    form_tag = match.group(0)
    assert 'data-sync-modal="true"' in form_tag
    assert f'data-sync-modal-label="{label}"' in form_tag


def assert_form_lacks_sync_modal_attrs(html: str, action_pattern: str) -> None:
    match = re.search(rf'<form[^>]*action="{action_pattern}"[^>]*>', html)
    assert match is not None
    form_tag = match.group(0)
    assert 'data-sync-modal="' not in form_tag
    assert 'data-sync-modal-label="' not in form_tag


def assert_form_absent(html: str, action_pattern: str) -> None:
    match = re.search(rf'<form[^>]*action="{action_pattern}"[^>]*>', html)
    assert match is None


def extract_primary_nav(html: str) -> str:
    match = re.search(
        r'<nav class="primary-nav" aria-label="Primary">(.+?)</nav>',
        html,
        re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def extract_primary_nav_links(html: str) -> list[dict[str, Any]]:
    nav_html = extract_primary_nav(html)
    anchors = re.findall(
        r'<a\s+class="primary-nav__link(?: is-active)?"\s+href="[^"]+"(?:\s+aria-current="page")?\s*>.+?</a>',
        nav_html,
        re.DOTALL,
    )
    assert anchors

    links: list[dict[str, Any]] = []
    for anchor in anchors:
        href_match = re.search(r'href="([^"]+)"', anchor)
        label_match = re.search(r">(.+?)</a>", anchor, re.DOTALL)
        class_match = re.search(r'class="([^"]+)"', anchor)
        assert href_match is not None
        assert label_match is not None
        assert class_match is not None
        links.append(
            {
                "href": href_match.group(1),
                "label": " ".join(label_match.group(1).split()),
                "is_active": "is-active" in class_match.group(1).split(),
                "has_aria_current": 'aria-current="page"' in anchor,
            }
        )
    return links


def assert_primary_nav_active_link(html: str, *, active_href: str, active_label: str) -> None:
    links = extract_primary_nav_links(html)
    active_links = [link for link in links if link["is_active"]]
    assert len(active_links) == 1
    active_link = active_links[0]
    assert active_link["href"] == active_href
    assert active_link["label"] == active_label
    assert active_link["has_aria_current"]

    for link in links:
        if link["href"] == active_href:
            continue
        assert not link["is_active"]
        assert not link["has_aria_current"]


def assert_docs_link_in_footer_not_primary_nav(html: str) -> None:
    nav_html = extract_primary_nav(html)
    assert 'href="/docs"' not in nav_html

    footer_match = re.search(r'<footer class="site-footer">(.+?)</footer>', html, re.DOTALL)
    assert footer_match is not None
    footer_html = footer_match.group(1)
    assert 'class="site-footer__docs-link"' in footer_html
    assert 'href="/docs"' in footer_html
    assert 'target="_blank"' in footer_html


def test_sync_modal_markup_and_form_attributes(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    settings_html = settings_response.text
    assert_sync_modal_markup(settings_html)
    assert 'action="/profiles/select"' in settings_html
    assert_form_absent(settings_html, r"/profiles/\d+/sync")
    assert_form_absent(settings_html, r"/profiles/\d+/sync-config")

    csrf_token = extract_csrf(settings_html)
    profile_id = create_profile(client, csrf_token, name="home")

    entities_response = client.get("/entities")
    assert entities_response.status_code == 200
    entities_html = entities_response.text
    assert_sync_modal_markup(entities_html)
    assert 'action="/profiles/select"' in entities_html
    assert_form_has_sync_modal_attrs(
        entities_html,
        r"/profiles/\d+/sync",
        "Syncing entities...",
    )
    assert_form_has_sync_modal_attrs(
        entities_html,
        r"/profiles/\d+/sync-config",
        "Syncing config items...",
    )

    config_items_response = client.get("/config-items")
    assert config_items_response.status_code == 200
    config_items_html = config_items_response.text
    assert_sync_modal_markup(config_items_html)
    assert 'action="/profiles/select"' in config_items_html
    assert_form_has_sync_modal_attrs(
        config_items_html,
        r"/profiles/\d+/sync-config",
        "Syncing config items...",
    )
    assert f"/profiles/{profile_id}/sync-config" in config_items_html


@pytest.mark.parametrize(
    ("path", "active_href", "active_label"),
    [
        ("/entities", "/entities", "Entities"),
        ("/config-items", "/config-items", "Config Items"),
        ("/suggestions", "/suggestions", "Automation Suggestions"),
        ("/entity-suggestions", "/entity-suggestions", "Entity Suggestions"),
        ("/automation-drafts", "/automation-drafts", "Automation Drafts"),
        ("/settings", "/settings", "Profiles"),
    ],
)
def test_primary_navigation_active_state_on_top_level_pages(
    client: TestClient,
    path: str,
    active_href: str,
    active_label: str,
) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert_primary_nav_active_link(response.text, active_href=active_href, active_label=active_label)
    assert_docs_link_in_footer_not_primary_nav(response.text)


def test_settings_sync_and_export_flow(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    response = client.get("/settings")
    assert response.status_code == 200

    csrf_token = extract_csrf(response.text)
    profile_id = create_profile(client, csrf_token, name="default")

    async def fake_test_connection(_: Any) -> dict[str, Any]:
        return {"version": "2026.2.0"}

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "light.kitchen",
                "state": "on",
                "attributes": {"friendly_name": "Kitchen Light"},
                "last_changed": "2026-02-15T00:00:00+00:00",
                "last_updated": "2026-02-15T00:01:00+00:00",
                "context": {"id": "abc"},
            },
            {
                "entity_id": "sensor.outdoor_temp",
                "state": "71",
                "attributes": {"unit_of_measurement": "F"},
                "last_changed": "2026-02-15T00:02:00+00:00",
                "last_updated": "2026-02-15T00:02:00+00:00",
                "context": {"id": "def"},
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [
                {
                    "area_id": "kitchen_area",
                    "name": "Kitchen",
                    "floor_id": "first_floor",
                }
            ],
            "devices": [
                {
                    "id": "device_kitchen_light",
                    "name_by_user": "Kitchen Ceiling",
                    "manufacturer": "Acme",
                    "model": "L-100",
                    "area_id": "kitchen_area",
                    "labels": ["label_room_kitchen"],
                }
            ],
            "entities": [
                {
                    "entity_id": "light.kitchen",
                    "device_id": "device_kitchen_light",
                    "labels": ["label_lighting"],
                    "platform": "hue",
                }
            ],
            "labels": [
                {"label_id": "label_lighting", "name": "Lighting"},
                {"label_id": "label_room_kitchen", "name": "Room - Kitchen"},
            ],
            "floors": [{"floor_id": "first_floor", "name": "First Floor"}],
        }

    monkeypatch.setattr("app.main.HAClient.test_connection", fake_test_connection)
    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_registry_metadata",
        fake_fetch_registry_metadata,
    )

    test_connection_response = client.post(
        f"/profiles/{profile_id}/test",
        data={
            "csrf_token": csrf_token,
            "next_url": "/settings",
        },
        follow_redirects=False,
    )
    assert test_connection_response.status_code == 303

    sync_response = client.post(
        f"/profiles/{profile_id}/sync",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entities?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert sync_response.status_code == 303

    entities_response = client.get(f"/entities?profile_id={profile_id}")
    assert entities_response.status_code == 200
    assert "light.kitchen" in entities_response.text
    assert "sensor.outdoor_temp" in entities_response.text
    assert "Kitchen" in entities_response.text
    assert "Kitchen (First Floor)" in entities_response.text

    detail_response = client.get(f"/entities/light.kitchen?profile_id={profile_id}")
    assert detail_response.status_code == 200
    assert "Kitchen Light" in detail_response.text
    assert "Kitchen Ceiling" in detail_response.text
    assert "Lighting" in detail_response.text
    assert_primary_nav_active_link(detail_response.text, active_href="/entities", active_label="Entities")
    assert_docs_link_in_footer_not_primary_nav(detail_response.text)

    export_json_response = client.get(f"/export/json?profile_id={profile_id}&q=light")
    assert export_json_response.status_code == 200
    exported = json.loads(export_json_response.text)
    assert len(exported) == 1
    assert exported[0]["entity_id"] == "light.kitchen"
    assert "pulled_at" in exported[0]
    assert exported[0]["area_name"] == "Kitchen"
    assert exported[0]["location_name"] == "Kitchen (First Floor)"
    assert exported[0]["device_name"] == "Kitchen Ceiling"
    assert exported[0]["labels"]["names"] == ["Lighting", "Room - Kitchen"]
    assert exported[0]["metadata"]["entity_platform"] == "hue"

    export_csv_response = client.get(f"/export/csv?profile_id={profile_id}&domain=light")
    assert export_csv_response.status_code == 200
    assert "text/csv" in export_csv_response.headers["content-type"]
    assert "light.kitchen" in export_csv_response.text
    assert "area_name" in export_csv_response.text
    assert "labels_json" in export_csv_response.text


def test_sync_config_items_list_and_detail_flow(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = client.get("/settings")
    assert response.status_code == 200

    csrf_token = extract_csrf(response.text)
    profile_id = create_profile(client, csrf_token, name="default")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "automation.evening_mode",
                "state": "on",
                "attributes": {"friendly_name": "Evening Mode", "id": "auto_evening_mode"},
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:01:00+00:00",
            },
            {
                "entity_id": "script.goodnight",
                "state": "off",
                "attributes": {"friendly_name": "Goodnight"},
                "last_changed": "2026-02-15T01:02:00+00:00",
                "last_updated": "2026-02-15T01:03:00+00:00",
            },
            {
                "entity_id": "scene.movie_time",
                "state": "2026-02-15T01:04:00+00:00",
                "attributes": {"friendly_name": "Movie Time", "id": "scene_movie_time"},
                "last_changed": "2026-02-15T01:04:00+00:00",
                "last_updated": "2026-02-15T01:04:00+00:00",
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [],
            "devices": [],
            "entities": [
                {"entity_id": "automation.evening_mode", "unique_id": "auto_evening_mode"},
                {"entity_id": "script.goodnight", "unique_id": "goodnight"},
                {"entity_id": "scene.movie_time", "unique_id": "scene_movie_time"},
            ],
            "labels": [],
            "floors": [],
        }

    async def fake_fetch_automation_config_ws(_: Any, entity_id: str) -> dict[str, Any]:
        assert entity_id == "automation.evening_mode"
        return {
            "alias": "Evening Mode",
            "description": "Turn lights on at sunset",
            "triggers": [{"trigger": "sun", "event": "sunset"}],
            "conditions": [{"condition": "state", "entity_id": "person.jason", "state": "home"}],
            "actions": [{"action": "light.turn_on", "target": {"entity_id": "light.living_room"}}],
        }

    async def fake_fetch_script_config_ws(_: Any, entity_id: str) -> dict[str, Any]:
        assert entity_id == "script.goodnight"
        return {
            "alias": "Goodnight",
            "description": "Night routine",
            "sequence": [
                {"action": "scene.turn_on", "target": {"entity_id": "scene.movie_time"}},
                {"action": "light.turn_off", "target": {"entity_id": ["light.kitchen"]}},
            ],
            "mode": "single",
        }

    async def fake_fetch_scene_config(_: Any, config_key: str) -> dict[str, Any]:
        assert config_key == "scene_movie_time"
        return {
            "name": "Movie Time",
            "entities": {
                "light.living_room": {"state": "off"},
                "media_player.tv": {"state": "on"},
            },
        }

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_automation_config_ws",
        fake_fetch_automation_config_ws,
    )
    monkeypatch.setattr(
        "app.main.HAClient.fetch_script_config_ws",
        fake_fetch_script_config_ws,
    )
    monkeypatch.setattr("app.main.HAClient.fetch_scene_config", fake_fetch_scene_config)

    sync_response = client.post(
        f"/profiles/{profile_id}/sync-config",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/config-items?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert sync_response.status_code == 303

    items_response = client.get(f"/config-items?profile_id={profile_id}")
    assert items_response.status_code == 200
    assert "automation.evening_mode" in items_response.text
    assert "script.goodnight" in items_response.text
    assert "scene.movie_time" in items_response.text

    scripts_only_response = client.get(f"/config-items?profile_id={profile_id}&kind=script")
    assert scripts_only_response.status_code == 200
    assert "script.goodnight" in scripts_only_response.text

    detail_match = re.search(r"/config-items/(\d+)\?[^\"']*config_sync_run_id=", scripts_only_response.text)
    assert detail_match is not None
    snapshot_id = int(detail_match.group(1))

    detail_response = client.get(
        f"/config-items/{snapshot_id}?profile_id={profile_id}"
    )
    assert detail_response.status_code == 200
    assert "Goodnight" in detail_response.text
    assert "sequence_count" in detail_response.text
    assert "scene.movie_time" in detail_response.text


def test_sync_config_items_partial_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    response = client.get("/settings")
    assert response.status_code == 200

    csrf_token = extract_csrf(response.text)
    profile_id = create_profile(client, csrf_token, name="default")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "automation.ok_rule",
                "state": "on",
                "attributes": {"friendly_name": "OK Rule", "id": "ok_rule"},
            },
            {
                "entity_id": "script.bad_rule",
                "state": "off",
                "attributes": {"friendly_name": "Bad Rule"},
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [],
            "devices": [],
            "entities": [
                {"entity_id": "automation.ok_rule", "unique_id": "ok_rule"},
                {"entity_id": "script.bad_rule", "unique_id": "bad_rule"},
            ],
            "labels": [],
            "floors": [],
        }

    async def fake_fetch_automation_config_ws(_: Any, entity_id: str) -> dict[str, Any]:
        assert entity_id == "automation.ok_rule"
        return {"alias": "OK Rule", "triggers": [], "actions": []}

    async def fake_fetch_script_config_ws(_: Any, entity_id: str) -> dict[str, Any]:
        assert entity_id == "script.bad_rule"
        raise HAClientError("ws failure")

    async def fake_fetch_script_config(_: Any, config_key: str) -> dict[str, Any]:
        assert config_key == "bad_rule"
        raise HAClientError("rest failure")

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_automation_config_ws",
        fake_fetch_automation_config_ws,
    )
    monkeypatch.setattr("app.main.HAClient.fetch_script_config_ws", fake_fetch_script_config_ws)
    monkeypatch.setattr("app.main.HAClient.fetch_script_config", fake_fetch_script_config)

    sync_response = client.post(
        f"/profiles/{profile_id}/sync-config",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/config-items?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert sync_response.status_code == 303

    items_response = client.get(f"/config-items?profile_id={profile_id}")
    assert items_response.status_code == 200
    assert "partial" in items_response.text

    error_filter_response = client.get(
        f"/config-items?profile_id={profile_id}&status=error"
    )
    assert error_filter_response.status_code == 200
    assert "script.bad_rule" in error_filter_response.text


def test_sync_config_items_missing_locator(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    response = client.get("/settings")
    assert response.status_code == 200

    csrf_token = extract_csrf(response.text)
    profile_id = create_profile(client, csrf_token, name="default")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "automation.ok_rule",
                "state": "on",
                "attributes": {"friendly_name": "OK Rule", "id": "ok_rule"},
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [],
            "devices": [],
            "entities": [
                {"entity_id": "automation.ok_rule", "unique_id": "ok_rule"},
                {"entity_id": "scene.no_locator"},
            ],
            "labels": [],
            "floors": [],
        }

    async def fake_fetch_automation_config_ws(_: Any, entity_id: str) -> dict[str, Any]:
        assert entity_id == "automation.ok_rule"
        return {"alias": "OK Rule", "triggers": [], "actions": []}

    async def fake_fetch_scene_config(_: Any, config_key: str) -> dict[str, Any]:
        raise AssertionError(f"scene config fetch should not be called without locator: {config_key}")

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_automation_config_ws",
        fake_fetch_automation_config_ws,
    )
    monkeypatch.setattr("app.main.HAClient.fetch_scene_config", fake_fetch_scene_config)

    sync_response = client.post(
        f"/profiles/{profile_id}/sync-config",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/config-items?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert sync_response.status_code == 303

    error_filter_response = client.get(
        f"/config-items?profile_id={profile_id}&status=error"
    )
    assert error_filter_response.status_code == 200
    assert "scene.no_locator" in error_filter_response.text
    assert "error" in error_filter_response.text

    detail_match = re.search(r"/config-items/(\d+)\?[^\"']*config_sync_run_id=", error_filter_response.text)
    assert detail_match is not None
    detail_response = client.get(
        f"/config-items/{detail_match.group(1)}?profile_id={profile_id}"
    )
    assert detail_response.status_code == 200
    assert "missing_config_locator" in detail_response.text


def test_empty_state_without_profiles(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    assert "No profiles configured yet." in settings_response.text
    assert "No enabled profiles" in settings_response.text

    entities_response = client.get("/entities")
    assert entities_response.status_code == 200
    assert "No profiles are configured yet." in entities_response.text
    assert "token setup" in entities_response.text

    config_items_response = client.get("/config-items")
    assert config_items_response.status_code == 200
    assert "No profiles are configured yet." in config_items_response.text
    assert "token setup" in config_items_response.text


def test_profile_switcher_session_and_query_precedence(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)

    alpha_id = create_profile(client, csrf_token, name="alpha")
    beta_id = create_profile(client, csrf_token, name="beta")

    entities_response = client.get("/entities")
    assert entities_response.status_code == 200
    assert "Active Profile:</strong> alpha" in entities_response.text

    switch_response = client.post(
        "/profiles/select",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(beta_id),
            "next_url": "/entities?q=light",
        },
        follow_redirects=False,
    )
    assert switch_response.status_code == 303
    location = switch_response.headers.get("location", "")
    assert "/entities" in location
    assert "q=light" in location
    assert f"profile_id={beta_id}" in location

    entities_after_switch = client.get("/entities")
    assert entities_after_switch.status_code == 200
    assert "Active Profile:</strong> beta" in entities_after_switch.text

    query_override = client.get(f"/entities?profile_id={alpha_id}")
    assert query_override.status_code == 200
    assert "Active Profile:</strong> alpha" in query_override.text

    entities_after_override = client.get("/entities")
    assert entities_after_override.status_code == 200
    assert "Active Profile:</strong> alpha" in entities_after_override.text


def test_disabled_profile_hidden_and_sync_blocked(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="home")

    disable_response = client.post(
        f"/profiles/{profile_id}/disable",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )
    assert disable_response.status_code == 303

    settings_after_disable = client.get("/settings")
    assert settings_after_disable.status_code == 200
    assert "Disabled" in settings_after_disable.text
    assert f"/profiles/{profile_id}/enable" in settings_after_disable.text
    assert f'<option value="{profile_id}"' not in settings_after_disable.text

    entities_after_disable = client.get("/entities")
    assert entities_after_disable.status_code == 200
    assert "No enabled profiles are available." in entities_after_disable.text

    async def fail_if_sync_called(_: Any) -> list[dict[str, Any]]:
        raise AssertionError("sync should not be called for disabled profile")

    monkeypatch.setattr("app.main.HAClient.fetch_states", fail_if_sync_called)
    sync_response = client.post(
        f"/profiles/{profile_id}/sync",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entities?profile_id={profile_id}",
        },
        follow_redirects=True,
    )
    assert sync_response.status_code == 200
    assert "is disabled. Re-enable it from settings." in sync_response.text


def test_disabling_active_profile_reassigns_active_profile(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    alpha_id = create_profile(client, csrf_token, name="alpha")
    beta_id = create_profile(client, csrf_token, name="beta")

    select_beta = client.post(
        "/profiles/select",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(beta_id),
            "next_url": "/entities",
        },
        follow_redirects=False,
    )
    assert select_beta.status_code == 303

    disable_beta = client.post(
        f"/profiles/{beta_id}/disable",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )
    assert disable_beta.status_code == 303

    entities_response = client.get("/entities")
    assert entities_response.status_code == 200
    assert "Active Profile:</strong> alpha" in entities_response.text
    assert f'<option value="{beta_id}"' not in entities_response.text
    assert f'<option value="{alpha_id}"' in entities_response.text


def test_llm_connection_and_automation_suggestion_run_flow(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="lab")

    create_connection = client.post(
        f"/profiles/{profile_id}/llm-connections",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
            "name": "Primary LLM",
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.1",
            "api_key_env_var": "TEST_LLM_KEY",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
            "is_enabled": "on",
            "is_default": "on",
        },
        follow_redirects=False,
    )
    assert create_connection.status_code == 303
    connection_id = get_llm_connection_id(profile_id, "Primary LLM")

    monkeypatch.setenv("TEST_LLM_KEY", "key-123")

    async def fake_chat_json(_: Any, __: str, ___: dict[str, Any]) -> dict[str, Any]:
        return {
            "suggestions": [
                {
                    "title": "Evening mode safety guard",
                    "summary": "Improve evening mode by adding occupancy and quiet-hour safeguards.",
                    "concept_type": "safety",
                    "target_kind": "existing_automation",
                    "target_entity_id": "automation.evening_mode",
                    "involved_entities": ["person.jason", "light.kitchen"],
                    "impact_score": 0.82,
                    "feasibility_score": 0.88,
                    "novelty_score": 0.41,
                    "confidence": 0.81,
                    "risk_level": "low",
                    "prerequisites": [
                        "Evening mode automation is enabled.",
                    ],
                    "verification_outline": [
                        "Run the automation manually in Home Assistant.",
                        "Confirm light.kitchen state changes as expected.",
                    ],
                    "rationale": "This reduces accidental triggers at night.",
                }
            ],
            "provider_debug": "token=key-123",
        }

    async def fake_test_connection(_: Any) -> dict[str, Any]:
        return {"status": "ok"}

    monkeypatch.setattr("app.main.OpenAICompatibleLLMClient.chat_json", fake_chat_json)
    monkeypatch.setattr("app.main.OpenAICompatibleLLMClient.test_connection", fake_test_connection)

    config_sync_run_id, snapshot_id = seed_config_snapshot(profile_id)

    queue_response = client.post(
        f"/profiles/{profile_id}/suggestions/runs",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/suggestions?profile_id={profile_id}",
            "llm_connection_id": str(connection_id),
            "config_sync_run_id": str(config_sync_run_id),
            "snapshot_id": str(snapshot_id),
        },
        follow_redirects=False,
    )
    assert queue_response.status_code == 303
    location = queue_response.headers.get("location", "")
    run_match = re.search(r"/suggestions/(\d+)\?", location)
    assert run_match is not None
    run_id = int(run_match.group(1))

    run_payload = wait_for_suggestion_run(client, profile_id, run_id)
    assert run_payload["run"]["status"] == "succeeded"
    assert run_payload["proposal_counts"].get("proposed", 0) >= 1
    proposal_id = int(run_payload["proposals"][0]["id"])

    with Session(db.get_engine()) as session:
        proposal = session.get(SuggestionProposal, proposal_id)
        assert proposal is not None
        assert proposal.raw_response_json is not None
        assert "key-123" not in proposal.raw_response_json
        assert "***redacted***" in proposal.raw_response_json

    accept_response = client.post(
        f"/suggestions/proposals/{proposal_id}/status",
        data={
            "csrf_token": csrf_token,
            "status": "accepted",
            "next_url": f"/suggestions/{run_id}?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert accept_response.status_code == 303

    detail_response = client.get(f"/suggestions/{run_id}?profile_id={profile_id}")
    assert detail_response.status_code == 200
    assert "automation.evening_mode" in detail_response.text
    assert "accepted" in detail_response.text

    test_connection_response = client.post(
        f"/llm-connections/{connection_id}/test",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
        },
        follow_redirects=True,
    )
    assert test_connection_response.status_code == 200
    assert "is reachable" in test_connection_response.text

    delete_response = client.post(
        f"/llm-connections/{connection_id}/delete",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
        },
        follow_redirects=True,
    )
    assert delete_response.status_code == 200
    assert "Primary LLM" not in delete_response.text


def test_missing_llm_key_and_disabled_connection_are_blocked(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="remote")

    create_connection = client.post(
        f"/profiles/{profile_id}/llm-connections",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
            "name": "Remote LLM",
            "base_url": "https://api.example.com/v1",
            "model": "gpt-4o-mini",
            "api_key_env_var": "MISSING_REMOTE_KEY",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
            "is_enabled": "on",
            "is_default": "on",
        },
        follow_redirects=False,
    )
    assert create_connection.status_code == 303
    connection_id = get_llm_connection_id(profile_id, "Remote LLM")

    monkeypatch.delenv("MISSING_REMOTE_KEY", raising=False)
    test_response = client.post(
        f"/llm-connections/{connection_id}/test",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
        },
        follow_redirects=True,
    )
    assert test_response.status_code == 200
    assert "LLM connection test failed" in test_response.text

    config_sync_run_id, snapshot_id = seed_config_snapshot(profile_id)
    disable_response = client.post(
        f"/llm-connections/{connection_id}/update",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
            "name": "Remote LLM",
            "base_url": "https://api.example.com/v1",
            "model": "gpt-4o-mini",
            "api_key_env_var": "MISSING_REMOTE_KEY",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
            # is_enabled omitted on purpose -> false
            "is_default": "on",
        },
        follow_redirects=False,
    )
    assert disable_response.status_code == 303

    queue_response = client.post(
        f"/profiles/{profile_id}/suggestions/runs",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/config-items?profile_id={profile_id}&config_sync_run_id={config_sync_run_id}",
            "llm_connection_id": str(connection_id),
            "snapshot_id": str(snapshot_id),
        },
        follow_redirects=True,
    )
    assert queue_response.status_code == 200
    assert "No enabled LLM connection configured for this profile." in queue_response.text


def test_suggestion_run_failure_exposes_debug_diagnostics(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="debuglab")

    create_connection = client.post(
        f"/profiles/{profile_id}/llm-connections",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
            "name": "Debug LLM",
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.1",
            "api_key_env_var": "TEST_LLM_KEY",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
            "is_enabled": "on",
            "is_default": "on",
        },
        follow_redirects=False,
    )
    assert create_connection.status_code == 303
    monkeypatch.setenv("TEST_LLM_KEY", "key-123")
    seed_config_snapshot(profile_id)

    async def fake_chat_json(_: Any, __: str, ___: dict[str, Any]) -> dict[str, Any]:
        raise LLMClientError("LLM message content is empty.")

    monkeypatch.setattr("app.main.OpenAICompatibleLLMClient.chat_json", fake_chat_json)

    queue_response = client.post(
        f"/profiles/{profile_id}/suggestions/runs",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/suggestions?profile_id={profile_id}",
            "idea_type": "obscure_automations",
            "mode": "obscure",
            "top_k": "2",
            "include_existing": "on",
            "include_new": "on",
        },
        follow_redirects=False,
    )
    assert queue_response.status_code == 303
    location = queue_response.headers.get("location", "")
    run_match = re.search(r"/suggestions/(\d+)\?", location)
    assert run_match is not None
    run_id = int(run_match.group(1))

    run_payload = wait_for_suggestion_run(client, profile_id, run_id)
    assert run_payload["run"]["status"] == "failed"
    debug_payload = run_payload["run"]["debug"]
    assert debug_payload["error"]["kind"] == "llm_empty_message"
    assert debug_payload["error"]["provider_message"] == "LLM message content is empty."
    assert "recommended_steps" in debug_payload["error"]
    assert debug_payload["llm_connection"]["model"] == "llama3.1"

    detail_response = client.get(f"/suggestions/{run_id}?profile_id={profile_id}")
    assert detail_response.status_code == 200
    assert "Debug Diagnostics" in detail_response.text
    assert "Recommended Next Steps" in detail_response.text
    assert "Run Audit Events" in detail_response.text
    assert "Copy Debug JSON" in detail_response.text
    assert "Copy Full Debug Report" in detail_response.text
    assert "Download Debug Report" in detail_response.text

    debug_report_response = client.get(f"/suggestions/{run_id}/debug.txt?profile_id={profile_id}")
    assert debug_report_response.status_code == 200
    assert debug_report_response.headers["content-type"].startswith("text/plain")
    report_text = debug_report_response.text
    assert "HA Entity Vault Suggestion Run Debug Report" in report_text
    assert "Debug JSON" in report_text
    assert "llm_empty_message" in report_text


def test_suggestion_run_failure_classifies_timeout(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="timeoutlab")

    create_connection = client.post(
        f"/profiles/{profile_id}/llm-connections",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
            "name": "Timeout LLM",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-5.2",
            "api_key_env_var": "TEST_LLM_KEY",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
            "is_enabled": "on",
            "is_default": "on",
        },
        follow_redirects=False,
    )
    assert create_connection.status_code == 303
    monkeypatch.setenv("TEST_LLM_KEY", "key-123")
    seed_config_snapshot(profile_id)

    async def fake_chat_json(_: Any, __: str, ___: dict[str, Any]) -> dict[str, Any]:
        raise LLMClientError(
            "Unable to reach LLM provider: ReadTimeout: request timed out (https://api.openai.com/v1/chat/completions)"
        )

    monkeypatch.setattr("app.main.OpenAICompatibleLLMClient.chat_json", fake_chat_json)

    queue_response = client.post(
        f"/profiles/{profile_id}/suggestions/runs",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/suggestions?profile_id={profile_id}",
            "idea_type": "obscure_automations",
            "mode": "obscure",
            "top_k": "2",
            "include_existing": "on",
            "include_new": "on",
        },
        follow_redirects=False,
    )
    assert queue_response.status_code == 303
    location = queue_response.headers.get("location", "")
    run_match = re.search(r"/suggestions/(\d+)\?", location)
    assert run_match is not None
    run_id = int(run_match.group(1))

    run_payload = wait_for_suggestion_run(client, profile_id, run_id)
    assert run_payload["run"]["status"] == "failed"
    debug_payload = run_payload["run"]["debug"]
    assert debug_payload["error"]["kind"] == "llm_timeout"
    assert "Increase timeout seconds" in " ".join(debug_payload["error"]["recommended_steps"])


def test_suggestions_page_includes_failed_runs_and_paginates(
    client: TestClient,
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="pagerlab")

    create_connection = client.post(
        f"/profiles/{profile_id}/llm-connections",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
            "name": "Pager LLM",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-5.2",
            "api_key_env_var": "TEST_LLM_KEY",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
            "is_enabled": "on",
            "is_default": "on",
        },
        follow_redirects=False,
    )
    assert create_connection.status_code == 303
    connection_id = get_llm_connection_id(profile_id, "Pager LLM")

    with Session(db.get_engine()) as session:
        for idx in range(12):
            run = SuggestionRun(
                profile_id=profile_id,
                llm_connection_id=connection_id,
                config_sync_run_id=None,
                run_kind="concept_v2",
                idea_type="general",
                custom_intent=None,
                mode="standard",
                top_k=10,
                include_existing=True,
                include_new=True,
                status="failed",
                target_count=0,
                processed_count=0,
                success_count=0,
                invalid_count=0,
                error_count=1,
                error=f"provider_error:run_{idx}",
                context_hash=None,
                filters_json=json.dumps({}),
                result_summary_json=json.dumps({"error_count": 1}),
                started_at=None,
                finished_at=None,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
            session.add(run)
            session.flush()
        session.commit()

    first_page = client.get(f"/suggestions?profile_id={profile_id}")
    assert first_page.status_code == 200
    assert "Showing 10 of 12 runs." in first_page.text
    assert "Page 1 of 2 (10 per page)" in first_page.text

    first_page_run_ids = [int(item) for item in re.findall(r'<td data-label="Run">#(\d+)</td>', first_page.text)]
    assert len(first_page_run_ids) == 10
    assert first_page_run_ids == sorted(first_page_run_ids, reverse=True)
    assert min(first_page_run_ids) > 2

    second_page = client.get(f"/suggestions?profile_id={profile_id}&page=2")
    assert second_page.status_code == 200
    assert "Page 2 of 2 (10 per page)" in second_page.text
    second_page_run_ids = [int(item) for item in re.findall(r'<td data-label="Run">#(\d+)</td>', second_page.text)]
    assert len(second_page_run_ids) == 2
    assert second_page_run_ids == sorted(second_page_run_ids, reverse=True)
    assert max(second_page_run_ids) < min(first_page_run_ids)


def test_suggestion_run_status_api_supports_lightweight_counts_mode(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="api-counts")
    connection_id = seed_default_llm_connection(profile_id, name="API Counts LLM")
    run_id, proposal_ids = seed_suggestion_run_with_proposals(profile_id, connection_id)

    lightweight_response = client.get(
        f"/api/suggestions/runs/{run_id}?profile_id={profile_id}&include_proposals=false"
    )
    assert lightweight_response.status_code == 200
    lightweight_payload = lightweight_response.json()
    assert lightweight_payload["run"]["id"] == run_id
    assert lightweight_payload["proposals"] == []
    assert lightweight_payload["proposal_counts"] == {
        "accepted": 1,
        "invalid": 1,
        "proposed": 1,
    }
    assert lightweight_payload["queue_counts"] == {
        "archived": 1,
        "queued": 1,
        "suggested": 1,
    }

    full_response = client.get(f"/api/suggestions/runs/{run_id}?profile_id={profile_id}")
    assert full_response.status_code == 200
    full_payload = full_response.json()
    assert sorted([int(item["id"]) for item in full_payload["proposals"]]) == sorted(proposal_ids)
    assert full_payload["proposal_counts"] == lightweight_payload["proposal_counts"]
    assert full_payload["queue_counts"] == lightweight_payload["queue_counts"]


def test_suggestions_page_renders_refresh_hooks(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="refresh-list")
    connection_id = seed_default_llm_connection(profile_id, name="Refresh List LLM")
    run_id, _ = seed_suggestion_run_with_proposals(profile_id, connection_id, run_status="running")

    response = client.get(f"/suggestions?profile_id={profile_id}")
    assert response.status_code == 200
    html = response.text
    assert 'src="/static/suggestion_run_refresh.js"' in html
    assert 'data-suggestion-refresh="list"' in html
    assert f'data-profile-id="{profile_id}"' in html
    assert 'data-status-filter=""' in html
    assert "data-refresh-banner" in html
    assert "data-refresh-now" in html
    assert f'data-run-id="{run_id}"' in html
    assert "data-run-status" in html
    assert "data-run-progress" in html
    assert "data-run-results" in html


def test_suggestion_run_detail_page_renders_refresh_hooks(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="refresh-detail")
    connection_id = seed_default_llm_connection(profile_id, name="Refresh Detail LLM")
    run_id, _ = seed_suggestion_run_with_proposals(profile_id, connection_id, run_status="running")

    response = client.get(f"/suggestions/{run_id}?profile_id={profile_id}")
    assert response.status_code == 200
    html = response.text
    assert 'data-suggestion-refresh="detail"' in html
    assert f'data-profile-id="{profile_id}"' in html
    assert f'data-run-id="{run_id}"' in html
    assert 'id="suggestion-run-status-chip"' in html
    assert "data-run-target" in html
    assert "data-run-processed" in html
    assert "data-run-success" in html
    assert "data-run-invalid" in html
    assert "data-run-error-count" in html
    assert "data-run-error" in html
    assert "data-run-started-at" in html
    assert "data-run-finished-at" in html
    assert "data-run-stage-counts" in html
    assert 'data-stage-count="suggested"' in html


def test_llm_presets_api_returns_expected_catalog(client: TestClient) -> None:
    response = client.get("/api/llm/presets")
    assert response.status_code == 200
    payload = response.json()
    assert "presets" in payload

    presets = payload["presets"]
    assert isinstance(presets, list)
    by_slug = {item["slug"]: item for item in presets}
    assert set(by_slug) == {"chatgpt", "claude", "gemini", "manual"}
    assert by_slug["chatgpt"]["default_api_key_env_var"] == "OPENAI_API_KEY"
    assert by_slug["claude"]["default_api_key_env_var"] == "ANTHROPIC_API_KEY"
    assert by_slug["gemini"]["default_api_key_env_var"] == "GOOGLE_API_KEY"
    assert "base_url" in by_slug["chatgpt"]["required_fields"]
    assert "model" in by_slug["chatgpt"]["required_fields"]
    assert "api_key_env_var" in by_slug["chatgpt"]["required_fields"]


def test_llm_models_api_discovers_provider_models(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)

    monkeypatch.setenv("OPENAI_API_KEY", "token-123")

    async def fake_list_models(_: Any) -> list[str]:
        return ["gpt-4o", "gpt-4o-mini"]

    monkeypatch.setattr("app.main.OpenAICompatibleLLMClient.list_models", fake_list_models)

    response = client.post(
        "/api/llm/models",
        data={
            "csrf_token": csrf_token,
            "preset_slug": "chatgpt",
            "base_url": "",
            "api_key_env_var": "",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["source"] == "provider"
    assert payload["resolved_base_url"] == "https://api.openai.com/v1"
    assert payload["resolved_api_key_env_var"] == "OPENAI_API_KEY"
    assert payload["models"] == ["gpt-4o", "gpt-4o-mini"]
    assert payload["default_model"] == "gpt-4o"
    assert payload["api_key_env_var_status"]["is_set"] is True


def test_llm_models_api_falls_back_when_discovery_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)

    monkeypatch.setenv("OPENAI_API_KEY", "token-123")

    async def fake_list_models(_: Any) -> list[str]:
        raise LLMClientError("provider unavailable")

    monkeypatch.setattr("app.main.OpenAICompatibleLLMClient.list_models", fake_list_models)

    response = client.post(
        "/api/llm/models",
        data={
            "csrf_token": csrf_token,
            "preset_slug": "chatgpt",
            "base_url": "",
            "api_key_env_var": "",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["source"] == "fallback"
    assert payload["models"]
    assert any("Model discovery failed" in warning for warning in payload["warnings"])
    assert payload["api_key_env_var_status"]["env_var"] == "OPENAI_API_KEY"
    assert payload["api_key_env_var_status"]["is_set"] is True


def test_llm_models_api_accepts_direct_api_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)

    async def fake_list_models(_: Any) -> list[str]:
        return ["gpt-4o-mini"]

    monkeypatch.setattr("app.main.OpenAICompatibleLLMClient.list_models", fake_list_models)

    response = client.post(
        "/api/llm/models",
        data={
            "csrf_token": csrf_token,
            "preset_slug": "chatgpt",
            "base_url": "https://api.openai.com/v1",
            "api_key_env_var": "sk-svcacct-REDACTED",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["source"] == "provider"
    assert payload["models"] == ["gpt-4o", "gpt-4o-mini"]
    assert payload["api_key_env_var_status"]["mode"] == "direct_key"
    assert payload["api_key_env_var_status"]["using_direct_key"] is True
    assert payload["api_key_env_var_status"]["env_var"] == ""
    assert payload["api_key_env_var_status"]["valid_name"] is False
    assert payload["api_key_env_var_status"]["looks_like_secret_value"] is True
    assert payload["warnings"] == []


def test_llm_test_draft_endpoint_success_and_validation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)

    monkeypatch.setenv("OPENAI_API_KEY", "token-123")

    async def fake_test_connection_verbose(_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "message": "reachable",
            "debug": {
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "attempts": [
                    {
                        "request_body": {"model": "gpt-4o"},
                        "response_status": 200,
                        "response_body": {"choices": [{"message": {"content": "{\"ok\":true}"}}]},
                    }
                ],
            },
        }

    monkeypatch.setattr("app.main.OpenAICompatibleLLMClient.test_connection_verbose", fake_test_connection_verbose)

    success_response = client.post(
        "/api/llm/test-draft",
        data={
            "csrf_token": csrf_token,
            "preset_slug": "chatgpt",
            "base_url": "",
            "model": "",
            "api_key_env_var": "",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
        },
    )
    assert success_response.status_code == 200
    success_payload = success_response.json()
    assert success_payload["ok"] is True
    assert "reachable" in success_payload["message"].lower()
    assert success_payload["debug"]["endpoint"].endswith("/chat/completions")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    missing_key_response = client.post(
        "/api/llm/test-draft",
        data={
            "csrf_token": csrf_token,
            "preset_slug": "chatgpt",
            "base_url": "",
            "model": "",
            "api_key_env_var": "",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
        },
    )
    assert missing_key_response.status_code == 200
    missing_key_payload = missing_key_response.json()
    assert missing_key_payload["ok"] is False
    assert any("OPENAI_API_KEY" in error for error in missing_key_payload["errors"])
    assert missing_key_payload["debug"] == {}

    raw_key_response = client.post(
        "/api/llm/test-draft",
        data={
            "csrf_token": csrf_token,
            "preset_slug": "manual",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "api_key_env_var": "sk-svcacct-REDACTED",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
        },
    )
    assert raw_key_response.status_code == 200
    raw_key_payload = raw_key_response.json()
    assert raw_key_payload["ok"] is True
    assert raw_key_payload["api_key_env_var_status"]["mode"] == "direct_key"
    assert raw_key_payload["debug"]["attempts"][0]["response_status"] == 200

    invalid_url_response = client.post(
        "/api/llm/test-draft",
        data={
            "csrf_token": csrf_token,
            "preset_slug": "manual",
            "base_url": "not-valid",
            "model": "gpt-4o-mini",
            "api_key_env_var": "OPENAI_API_KEY",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
        },
    )
    assert invalid_url_response.status_code == 200
    invalid_url_payload = invalid_url_response.json()
    assert invalid_url_payload["ok"] is False
    assert "http:// or https://" in " ".join(invalid_url_payload["errors"])


def test_llm_connection_create_and_update_hydrate_preset_defaults(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="preset-hydration")

    create_response = client.post(
        f"/profiles/{profile_id}/llm-connections",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
            "name": "Preset Defaults",
            "preset_slug": "chatgpt",
            "base_url": "",
            "model": "",
            "api_key_env_var": "",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
            "is_enabled": "on",
            "is_default": "on",
        },
        follow_redirects=False,
    )
    assert create_response.status_code == 303
    connection_id = get_llm_connection_id(profile_id, "Preset Defaults")

    with Session(db.get_engine()) as session:
        created = session.get(LLMConnection, connection_id)
        assert created is not None
        assert created.base_url == "https://api.openai.com/v1"
        assert created.model == "gpt-4o"
        assert created.api_key_env_var == "OPENAI_API_KEY"

    update_response = client.post(
        f"/llm-connections/{connection_id}/update",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
            "name": "Preset Defaults",
            "preset_slug": "gemini",
            "base_url": "",
            "model": "",
            "api_key_env_var": "",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
            "is_enabled": "on",
            "is_default": "on",
        },
        follow_redirects=False,
    )
    assert update_response.status_code == 303

    with Session(db.get_engine()) as session:
        updated = session.get(LLMConnection, connection_id)
        assert updated is not None
        assert updated.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
        assert updated.model == "gemini-2.5-pro"
        assert updated.api_key_env_var == "GOOGLE_API_KEY"


def test_config_items_shows_suggest_for_successful_automation_only(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="buttons")

    create_connection = client.post(
        f"/profiles/{profile_id}/llm-connections",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/settings?profile_id={profile_id}",
            "name": "Buttons LLM",
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.1",
            "api_key_env_var": "",
            "timeout_seconds": "20",
            "temperature": "0.2",
            "max_output_tokens": "900",
            "extra_headers_json": "",
            "is_enabled": "on",
            "is_default": "on",
        },
        follow_redirects=False,
    )
    assert create_connection.status_code == 303

    with Session(db.get_engine()) as session:
        run = ConfigSyncRun(
            profile_id=profile_id,
            pulled_at=utcnow(),
            item_count=2,
            success_count=1,
            error_count=1,
            duration_ms=1,
            status="partial",
            error=None,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        assert run.id is not None

        auto = ConfigSnapshot(
            profile_id=profile_id,
            config_sync_run_id=run.id,
            kind="automation",
            entity_id="automation.ok_rule",
            config_key="ok_rule",
            name="OK Rule",
            state="on",
            fetch_status="success",
            fetch_error=None,
            summary_json=json.dumps({"name": "OK Rule"}),
            references_json=json.dumps({"entity_id": ["person.jason"]}),
            config_json=json.dumps({"alias": "OK Rule", "trigger": [], "action": []}),
            attributes_json=json.dumps({}),
            metadata_json=json.dumps({}),
            pulled_at=run.pulled_at,
        )
        script = ConfigSnapshot(
            profile_id=profile_id,
            config_sync_run_id=run.id,
            kind="script",
            entity_id="script.bad_rule",
            config_key="bad_rule",
            name="Bad Rule",
            state="off",
            fetch_status="error",
            fetch_error="failed",
            summary_json=json.dumps({"name": "Bad Rule"}),
            references_json=json.dumps({"entity_id": ["light.kitchen"]}),
            config_json=None,
            attributes_json=json.dumps({}),
            metadata_json=json.dumps({}),
            pulled_at=run.pulled_at,
        )
        session.add(auto)
        session.add(script)
        session.commit()
        session.refresh(auto)
        session.refresh(script)
        assert auto.id is not None
        assert script.id is not None
        run_id = run.id
        auto_snapshot_id = auto.id
        script_snapshot_id = script.id

    response = client.get(f"/config-items?profile_id={profile_id}&config_sync_run_id={run_id}")
    assert response.status_code == 200
    assert f'name="snapshot_id" value="{auto_snapshot_id}"' in response.text
    assert f'name="snapshot_id" value="{script_snapshot_id}"' not in response.text


def test_run_entity_suggestions_and_readiness_api(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="home")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.living_temp",
                "state": "72",
                "attributes": {
                    "friendly_name": "Living Room Temperature",
                    "device_class": "temperature",
                    "state_class": "measurement",
                    "unit_of_measurement": "F",
                },
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:00:00+00:00",
            },
            {
                "entity_id": "event.scene_button",
                "state": "2026-02-15T01:00:00+00:00",
                "attributes": {"friendly_name": "Scene Button", "event_type": "KeyPressed"},
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:00:00+00:00",
            },
            {
                "entity_id": "light.unassigned",
                "state": "on",
                "attributes": {"friendly_name": "Unassigned Light"},
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:00:00+00:00",
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [{"area_id": "living_room", "name": "Living Room"}],
            "devices": [
                {
                    "id": "dev_temp",
                    "name_by_user": "Living Temp Device",
                    "area_id": "living_room",
                    "labels": ["label_climate"],
                }
            ],
            "entities": [
                {
                    "entity_id": "sensor.living_temp",
                    "device_id": "dev_temp",
                    "labels": ["label_climate"],
                }
            ],
            "labels": [{"label_id": "label_climate", "name": "Climate"}],
            "floors": [],
        }

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    sync_response = client.post(
        f"/profiles/{profile_id}/sync",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entities?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert sync_response.status_code == 303

    suggestion_response = client.post(
        f"/profiles/{profile_id}/run-entity-suggestions",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entity-suggestions?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert suggestion_response.status_code == 303

    suggestions_page = client.get(f"/entity-suggestions?profile_id={profile_id}")
    assert suggestions_page.status_code == 200
    assert "sensor.living_temp" in suggestions_page.text
    assert "light.unassigned" not in suggestions_page.text
    assert "event.scene_button" not in suggestions_page.text
    assert "Run Summary (run-wide):" in suggestions_page.text
    assert "Candidate Scope:" in suggestions_page.text
    assert "sensor, binary_sensor, lock" in suggestions_page.text

    api_response = client.get(f"/api/entity-suggestions?profile_id={profile_id}")
    assert api_response.status_code == 200
    payload = api_response.json()
    assert payload["total"] == 1
    assert payload["suggestion_run"] is not None

    by_entity = {item["entity_id"]: item for item in payload["items"]}
    assert by_entity["sensor.living_temp"]["readiness_status"] == "ready"
    assert "light.unassigned" not in by_entity
    assert "event.scene_button" not in by_entity


def test_run_entity_suggestions_downgrades_area_when_enrichment_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="no-enrichment")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.office_temp",
                "state": "71",
                "attributes": {
                    "friendly_name": "Office Temperature",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:00:00+00:00",
            },
            {
                "entity_id": "binary_sensor.office_motion",
                "state": "off",
                "attributes": {
                    "friendly_name": "Office Motion",
                    "device_class": "motion",
                },
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:00:00+00:00",
            },
            {
                "entity_id": "lock.front_door",
                "state": "locked",
                "attributes": {"friendly_name": "Front Door Lock"},
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:00:00+00:00",
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"areas": [], "devices": [], "entities": [], "labels": [], "floors": []}

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    sync_response = client.post(
        f"/profiles/{profile_id}/sync",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entities?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert sync_response.status_code == 303

    suggestion_response = client.post(
        f"/profiles/{profile_id}/run-entity-suggestions",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entity-suggestions?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert suggestion_response.status_code == 303

    suggestions_page = client.get(f"/entity-suggestions?profile_id={profile_id}")
    assert suggestions_page.status_code == 200
    assert "Area checks were downgraded for this run because no area/device enrichment was available during scoring." in suggestions_page.text

    api_response = client.get(f"/api/entity-suggestions?profile_id={profile_id}")
    assert api_response.status_code == 200
    payload = api_response.json()
    assert payload["total"] == 3
    assert payload["suggestion_run"]["blocked_count"] == 0
    assert payload["suggestion_run"]["needs_review_count"] == 3

    for item in payload["items"]:
        assert item["readiness_status"] == "needs_review"
        issue_codes = {issue["code"] for issue in item["issues"]}
        assert "missing_area_enrichment_unavailable" in issue_codes
        assert "missing_area" not in issue_codes


def test_run_entity_suggestions_keeps_strict_area_when_any_enrichment_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="strict-enrichment")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.good_temp",
                "state": "70",
                "attributes": {
                    "friendly_name": "Good Temperature",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:00:00+00:00",
            },
            {
                "entity_id": "sensor.missing_area_temp",
                "state": "69",
                "attributes": {
                    "friendly_name": "Missing Area Temperature",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:00:00+00:00",
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [{"area_id": "living_room", "name": "Living Room"}],
            "devices": [
                {
                    "id": "dev_good",
                    "name_by_user": "Good Sensor Device",
                    "area_id": "living_room",
                    "labels": ["label_climate"],
                }
            ],
            "entities": [
                {
                    "entity_id": "sensor.good_temp",
                    "device_id": "dev_good",
                    "labels": ["label_climate"],
                }
            ],
            "labels": [{"label_id": "label_climate", "name": "Climate"}],
            "floors": [],
        }

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    sync_response = client.post(
        f"/profiles/{profile_id}/sync",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entities?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert sync_response.status_code == 303

    suggestion_response = client.post(
        f"/profiles/{profile_id}/run-entity-suggestions",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entity-suggestions?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert suggestion_response.status_code == 303

    suggestions_page = client.get(f"/entity-suggestions?profile_id={profile_id}")
    assert suggestions_page.status_code == 200
    assert "Area checks were downgraded for this run because no area/device enrichment was available during scoring." not in suggestions_page.text

    api_response = client.get(f"/api/entity-suggestions?profile_id={profile_id}")
    assert api_response.status_code == 200
    payload = api_response.json()
    by_entity = {item["entity_id"]: item for item in payload["items"]}
    assert by_entity["sensor.missing_area_temp"]["readiness_status"] == "blocked"
    missing_area_codes = {issue["code"] for issue in by_entity["sensor.missing_area_temp"]["issues"]}
    assert "missing_area" in missing_area_codes
    assert "missing_area_enrichment_unavailable" not in missing_area_codes


def test_workflow_queue_only_includes_fixable_suggestions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-queue")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.fixable_area",
                "state": "70",
                "attributes": {
                    "friendly_name": "Fixable Area Sensor",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            },
            {
                "entity_id": "sensor.manual_only",
                "state": "72",
                "attributes": {
                    "friendly_name": "Manual Only Sensor",
                    "device_class": "temperature",
                    "state_class": "measurement",
                    "area_id": "office_area",
                },
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [{"area_id": "office_area", "name": "Office"}],
            "devices": [],
            "entities": [
                {
                    "entity_id": "sensor.manual_only",
                    "labels": ["label_existing"],
                }
            ],
            "labels": [{"label_id": "label_existing", "name": "Existing"}],
            "floors": [],
        }

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)

    queue_response = client.get(f"/entity-suggestions/workflow?profile_id={profile_id}")
    assert queue_response.status_code == 200
    assert "sensor.fixable_area" in queue_response.text
    assert "sensor.manual_only" not in queue_response.text


def test_workflow_detail_uses_registry_areas_dropdown_with_create_option(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-area-dropdown")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.office_temp",
                "state": "70",
                "attributes": {
                    "friendly_name": "Office Temp",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    sync_metadata_calls = {"count": 0}

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        sync_metadata_calls["count"] += 1
        return {
            "areas": [
                {"area_id": "office_area", "name": "Office"},
                {"area_id": "garage_area", "name": "Garage"},
            ],
            "devices": [],
            "entities": [],
            "labels": [],
            "floors": [],
        }

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    assert sync_metadata_calls["count"] == 1

    async def fail_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        raise AssertionError("workflow detail should not call fetch_registry_metadata")

    area_registry_calls = {"count": 0}

    async def fake_fetch_area_registry_entries(_: Any) -> list[dict[str, Any]]:
        area_registry_calls["count"] += 1
        return [
            {"area_id": "office_area", "name": "Office"},
            {"area_id": "garage_area", "name": "Garage"},
        ]

    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fail_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_area_registry_entries",
        fake_fetch_area_registry_entries,
    )

    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    suggestion_id = suggestions_payload["items"][0]["id"]
    detail_response = client.get(
        f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}"
    )
    assert detail_response.status_code == 200
    assert area_registry_calls["count"] == 1
    assert 'option value="__create_new_area__"' in detail_response.text
    assert 'option value="office_area">Office</option>' in detail_response.text
    assert 'option value="garage_area">Garage</option>' in detail_response.text


def test_workflow_detail_area_option_adds_parent_device_for_entity_named_area(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-area-parent-device-label")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.missing_area_target",
                "state": "70",
                "attributes": {
                    "friendly_name": "2024 IONIQ 5 Air Conditioner",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [],
            "devices": [],
            "entities": [],
            "labels": [],
            "floors": [],
        }

    async def fake_fetch_area_registry_entries(_: Any) -> list[dict[str, Any]]:
        return [
            {"area_id": "office_area", "name": "Office"},
            {"area_id": "on_the_go", "name": "On the go"},
        ]

    async def fake_fetch_entity_registry_entries(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.missing_area_target",
                "name": "2024 IONIQ 5 Air Conditioner",
                "device_id": "dev_ioniq_5",
                "area_id": "on_the_go",
            }
        ]

    async def fake_fetch_device_registry_entries(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "dev_ioniq_5",
                "name_by_user": "2024 IONIQ 5 (IONIQ 5)",
                "area_id": "on_the_go",
            }
        ]

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_area_registry_entries",
        fake_fetch_area_registry_entries,
    )
    monkeypatch.setattr(
        "app.main.HAClient.fetch_entity_registry_entries",
        fake_fetch_entity_registry_entries,
    )
    monkeypatch.setattr(
        "app.main.HAClient.fetch_device_registry_entries",
        fake_fetch_device_registry_entries,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    by_entity = {item["entity_id"]: item for item in suggestions_payload["items"]}
    suggestion_id = by_entity["sensor.missing_area_target"]["id"]
    detail_response = client.get(
        f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}"
    )
    assert detail_response.status_code == 200
    assert (
        'option value="on_the_go">On the go (Parent device: 2024 IONIQ 5 (IONIQ 5))</option>'
        in detail_response.text
    )
    parent_area_option_index = detail_response.text.find(
        'option value="on_the_go">On the go (Parent device: 2024 IONIQ 5 (IONIQ 5))</option>'
    )
    office_area_option_index = detail_response.text.find('option value="office_area">Office</option>')
    assert parent_area_option_index >= 0
    assert office_area_option_index >= 0
    assert parent_area_option_index < office_area_option_index
    assert "Parent Device:" in detail_response.text
    assert "2024 IONIQ 5 (IONIQ 5)" in detail_response.text
    assert "Assigned Area:" in detail_response.text
    assert "On the go" in detail_response.text


def test_workflow_detail_registry_fetches_use_ttl_cache(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-registry-cache-ttl")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.cache_target",
                "state": "70",
                "attributes": {
                    "friendly_name": "Cache Target",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"areas": [], "devices": [], "entities": [], "labels": [], "floors": []}

    async def fake_fetch_area_registry_entries(_: Any) -> list[dict[str, Any]]:
        return [{"area_id": "office_area", "name": "Office"}]

    entity_registry_calls = {"count": 0}
    device_registry_calls = {"count": 0}

    async def fake_fetch_entity_registry_entries(_: Any) -> list[dict[str, Any]]:
        entity_registry_calls["count"] += 1
        return [{"entity_id": "sensor.cache_target", "device_id": "device_cache_target"}]

    async def fake_fetch_device_registry_entries(_: Any) -> list[dict[str, Any]]:
        device_registry_calls["count"] += 1
        return [{"id": "device_cache_target", "name_by_user": "Cache Device", "area_id": "office_area"}]

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_area_registry_entries",
        fake_fetch_area_registry_entries,
    )
    monkeypatch.setattr(
        "app.main.HAClient.fetch_entity_registry_entries",
        fake_fetch_entity_registry_entries,
    )
    monkeypatch.setattr(
        "app.main.HAClient.fetch_device_registry_entries",
        fake_fetch_device_registry_entries,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    suggestion_id = suggestions_payload["items"][0]["id"]

    monotonic_clock = {"now": 100.0}

    def fake_perf_counter() -> float:
        return monotonic_clock["now"]

    monkeypatch.setattr("app.main.perf_counter", fake_perf_counter)
    monkeypatch.setattr("app.main.WORKFLOW_REGISTRY_CACHE_TTL_SECONDS", 10.0)

    first_response = client.get(f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}")
    assert first_response.status_code == 200
    assert entity_registry_calls["count"] == 1
    assert device_registry_calls["count"] == 1

    monotonic_clock["now"] = 105.0
    second_response = client.get(f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}")
    assert second_response.status_code == 200
    assert entity_registry_calls["count"] == 1
    assert device_registry_calls["count"] == 1

    monotonic_clock["now"] = 111.0
    third_response = client.get(f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}")
    assert third_response.status_code == 200
    assert entity_registry_calls["count"] == 2
    assert device_registry_calls["count"] == 2


def test_workflow_detail_places_apply_form_before_context_and_issues(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-layout-order")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.order_target",
                "state": "70",
                "attributes": {
                    "friendly_name": "Order Target",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"areas": [], "devices": [], "entities": [], "labels": [], "floors": []}

    async def fake_fetch_area_registry_entries(_: Any) -> list[dict[str, Any]]:
        return [{"area_id": "office_area", "name": "Office"}]

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_area_registry_entries",
        fake_fetch_area_registry_entries,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    suggestion_id = suggestions_payload["items"][0]["id"]
    detail_response = client.get(
        f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}"
    )
    assert detail_response.status_code == 200
    apply_form_index = detail_response.text.find(
        f'action="/entity-suggestions/{suggestion_id}/workflow/apply"'
    )
    context_heading_index = detail_response.text.find("<h3>Entity Context</h3>")
    fixable_heading_index = detail_response.text.find("<h3>Fixable Issues</h3>")
    assert apply_form_index >= 0
    assert context_heading_index >= 0
    assert fixable_heading_index >= 0
    assert apply_form_index < context_heading_index
    assert apply_form_index < fixable_heading_index


def test_workflow_detail_shows_required_resolution_guidance(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-required-guidance")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.required_target",
                "state": "70",
                "attributes": {
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"areas": [], "devices": [], "entities": [], "labels": [], "floors": []}

    async def fake_fetch_area_registry_entries(_: Any) -> list[dict[str, Any]]:
        return [{"area_id": "office_area", "name": "Office"}]

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_area_registry_entries",
        fake_fetch_area_registry_entries,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    suggestion_id = suggestions_payload["items"][0]["id"]
    detail_response = client.get(
        f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}"
    )
    assert detail_response.status_code == 200
    assert "Required to resolve naming issue" in detail_response.text
    assert "Choose an existing area or create a new one to resolve area mapping." in detail_response.text
    assert "Required when Create New Area is selected to resolve area mapping" in detail_response.text


def test_workflow_detail_non_required_fields_remain_editable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-non-required-editable")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.editable_target",
                "state": "70",
                "attributes": {
                    "friendly_name": "Editable Target",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"areas": [], "devices": [], "entities": [], "labels": [], "floors": []}

    async def fake_fetch_area_registry_entries(_: Any) -> list[dict[str, Any]]:
        return [{"area_id": "office_area", "name": "Office"}]

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_area_registry_entries",
        fake_fetch_area_registry_entries,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    suggestion_id = suggestions_payload["items"][0]["id"]
    detail_response = client.get(
        f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}"
    )
    assert detail_response.status_code == 200

    friendly_input = re.search(r'<input[^>]*name="friendly_name"[^>]*>', detail_response.text)
    assert friendly_input is not None
    assert "disabled" not in friendly_input.group(0)

    area_select = re.search(r'<select[^>]*name="area_id"[^>]*>', detail_response.text)
    assert area_select is not None
    assert "disabled" not in area_select.group(0)

    device_class_input = re.search(r'<input[^>]*name="device_class"[^>]*>', detail_response.text)
    assert device_class_input is not None
    assert "disabled" not in device_class_input.group(0)

    labels_input = re.search(r'<input[^>]*name="labels_csv"[^>]*>', detail_response.text)
    assert labels_input is not None
    assert "disabled" not in labels_input.group(0)
    assert "Required to resolve naming issue" not in detail_response.text


def test_workflow_detail_area_dropdown_falls_back_to_snapshot_on_ha_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-area-fallback")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.target_missing_area",
                "state": "70",
                "attributes": {
                    "friendly_name": "Target Missing Area",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            },
            {
                "entity_id": "sensor.snapshot_area_source",
                "state": "71",
                "attributes": {
                    "friendly_name": "Snapshot Area Source",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [{"area_id": "office_area", "name": "Office"}],
            "devices": [],
            "entities": [{"entity_id": "sensor.snapshot_area_source", "area_id": "office_area"}],
            "labels": [],
            "floors": [],
        }

    async def fake_fetch_area_registry_entries(_: Any) -> list[dict[str, Any]]:
        raise HAClientError(
            "Unable to reach Home Assistant WebSocket API: sent 1009 frame exceeds limit"
        )

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_area_registry_entries",
        fake_fetch_area_registry_entries,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    by_entity = {item["entity_id"]: item for item in suggestions_payload["items"]}
    suggestion_id = by_entity["sensor.target_missing_area"]["id"]

    detail_response = client.get(
        f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}"
    )
    assert detail_response.status_code == 200
    assert "Area options source: current suggestion run snapshot." in detail_response.text
    assert "Unable to refresh from Home Assistant: Unable to reach Home Assistant WebSocket API: sent 1009 frame exceeds limit." in detail_response.text
    assert 'option value="office_area">Office</option>' in detail_response.text


def test_workflow_apply_friendly_name_and_existing_area(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-apply")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.office_temp",
                "state": "69",
                "attributes": {
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"areas": [], "devices": [], "entities": [], "labels": [], "floors": []}

    applied: dict[str, Any] = {}

    async def fake_update_entity_registry_entry(_: Any, **payload: Any) -> dict[str, Any]:
        applied.update(payload)
        return {"entity_id": payload["entity_id"]}

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.update_entity_registry_entry",
        fake_update_entity_registry_entry,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    suggestion_id = suggestions_payload["items"][0]["id"]

    apply_response = client.post(
        f"/entity-suggestions/{suggestion_id}/workflow/apply",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}",
            "friendly_name": "Office Temperature",
            "area_id": "office_area",
            "new_area_name": "",
            "device_class": "",
            "labels_csv": "",
        },
        follow_redirects=False,
    )
    assert apply_response.status_code == 303
    assert applied["entity_id"] == "sensor.office_temp"
    assert applied["name"] == "Office Temperature"
    assert applied["area_id"] == "office_area"

    detail_payload = client.get(
        f"/api/entity-suggestions/{suggestion_id}?profile_id={profile_id}"
    ).json()
    assert detail_payload["workflow_status"] == "applied_pending_recheck"
    assert detail_payload["workflow_updated_at"] is not None


def test_workflow_apply_allows_non_issue_field_when_workflow_eligible(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-non-issue-field")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.non_issue_field_target",
                "state": "69",
                "attributes": {
                    "friendly_name": "Original Name",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"areas": [], "devices": [], "entities": [], "labels": [], "floors": []}

    applied: dict[str, Any] = {}

    async def fake_update_entity_registry_entry(_: Any, **payload: Any) -> dict[str, Any]:
        applied.update(payload)
        return {"entity_id": payload["entity_id"]}

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.update_entity_registry_entry",
        fake_update_entity_registry_entry,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    suggestion_id = suggestions_payload["items"][0]["id"]

    apply_response = client.post(
        f"/entity-suggestions/{suggestion_id}/workflow/apply",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}",
            "friendly_name": "Updated Name",
            "area_id": "",
            "new_area_name": "",
            "device_class": "",
            "labels_csv": "",
        },
        follow_redirects=False,
    )
    assert apply_response.status_code == 303
    assert applied["entity_id"] == "sensor.non_issue_field_target"
    assert applied["name"] == "Updated Name"
    assert "area_id" not in applied

    detail_payload = client.get(
        f"/api/entity-suggestions/{suggestion_id}?profile_id={profile_id}"
    ).json()
    assert detail_payload["workflow_status"] == "applied_pending_recheck"


def test_workflow_apply_creates_new_area_when_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-new-area")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.garage_temp",
                "state": "66",
                "attributes": {
                    "friendly_name": "Garage Temperature",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"areas": [{"area_id": "kitchen_area", "name": "Kitchen"}], "devices": [], "entities": [], "labels": [], "floors": []}

    async def fake_fetch_area_registry_entries(_: Any) -> list[dict[str, Any]]:
        return [{"area_id": "kitchen_area", "name": "Kitchen"}]

    async def fake_create_area_registry_entry(_: Any, *, name: str) -> dict[str, Any]:
        assert name == "Garage"
        return {"area_id": "garage_area", "name": "Garage"}

    applied: dict[str, Any] = {}

    async def fake_update_entity_registry_entry(_: Any, **payload: Any) -> dict[str, Any]:
        applied.update(payload)
        return {"entity_id": payload["entity_id"]}

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_area_registry_entries",
        fake_fetch_area_registry_entries,
    )
    monkeypatch.setattr(
        "app.main.HAClient.create_area_registry_entry",
        fake_create_area_registry_entry,
    )
    monkeypatch.setattr(
        "app.main.HAClient.update_entity_registry_entry",
        fake_update_entity_registry_entry,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    suggestion_id = suggestions_payload["items"][0]["id"]

    apply_response = client.post(
        f"/entity-suggestions/{suggestion_id}/workflow/apply",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}",
            "friendly_name": "",
            "area_id": "",
            "new_area_name": "Garage",
            "device_class": "",
            "labels_csv": "",
        },
        follow_redirects=False,
    )
    assert apply_response.status_code == 303
    assert applied["area_id"] == "garage_area"

    with Session(db.get_engine()) as session:
        row = session.get(EntitySuggestion, suggestion_id)
        assert row is not None
        result_payload = json.loads(row.workflow_result_json or "{}")
        assert result_payload["area_resolution"]["mode"] == "created"


def test_workflow_apply_reuses_existing_area_by_name_with_areas_only_lookup(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-existing-area-by-name")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.garage_temp",
                "state": "66",
                "attributes": {
                    "friendly_name": "Garage Temperature",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"areas": [], "devices": [], "entities": [], "labels": [], "floors": []}

    async def fake_fetch_area_registry_entries(_: Any) -> list[dict[str, Any]]:
        return [{"area_id": "garage_area", "name": "Garage"}]

    async def fail_create_area_registry_entry(_: Any, *, name: str) -> dict[str, Any]:
        raise AssertionError(f"create_area_registry_entry should not be called: {name}")

    applied: dict[str, Any] = {}

    async def fake_update_entity_registry_entry(_: Any, **payload: Any) -> dict[str, Any]:
        applied.update(payload)
        return {"entity_id": payload["entity_id"]}

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.fetch_area_registry_entries",
        fake_fetch_area_registry_entries,
    )
    monkeypatch.setattr(
        "app.main.HAClient.create_area_registry_entry",
        fail_create_area_registry_entry,
    )
    monkeypatch.setattr(
        "app.main.HAClient.update_entity_registry_entry",
        fake_update_entity_registry_entry,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    suggestion_id = suggestions_payload["items"][0]["id"]

    apply_response = client.post(
        f"/entity-suggestions/{suggestion_id}/workflow/apply",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/entity-suggestions/{suggestion_id}/workflow?profile_id={profile_id}",
            "friendly_name": "",
            "area_id": "",
            "new_area_name": "garage",
            "device_class": "",
            "labels_csv": "",
        },
        follow_redirects=False,
    )
    assert apply_response.status_code == 303
    assert applied["area_id"] == "garage_area"

    with Session(db.get_engine()) as session:
        row = session.get(EntitySuggestion, suggestion_id)
        assert row is not None
        result_payload = json.loads(row.workflow_result_json or "{}")
        assert result_payload["area_resolution"]["mode"] == "existing_by_name"
        assert result_payload["area_resolution"]["area_id"] == "garage_area"


def test_workflow_apply_device_class_and_labels(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-device-label")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.semantic_missing",
                "state": "1",
                "attributes": {"friendly_name": "Semantic Missing", "area_id": "office_area"},
            },
            {
                "entity_id": "sensor.labels_missing",
                "state": "22",
                "attributes": {
                    "friendly_name": "Labels Missing",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [{"area_id": "office_area", "name": "Office"}],
            "devices": [
                {
                    "id": "dev_labels",
                    "name_by_user": "Label Sensor Device",
                    "area_id": "office_area",
                }
            ],
            "entities": [
                {"entity_id": "sensor.semantic_missing", "area_id": "office_area"},
                {"entity_id": "sensor.labels_missing", "device_id": "dev_labels"},
            ],
            "labels": [],
            "floors": [],
        }

    updates: list[dict[str, Any]] = []

    async def fake_update_entity_registry_entry(_: Any, **payload: Any) -> dict[str, Any]:
        updates.append(payload)
        return {"entity_id": payload["entity_id"]}

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.HAClient.update_entity_registry_entry",
        fake_update_entity_registry_entry,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    by_entity = {item["entity_id"]: item for item in suggestions_payload["items"]}

    semantic_id = by_entity["sensor.semantic_missing"]["id"]
    labels_id = by_entity["sensor.labels_missing"]["id"]

    semantic_apply = client.post(
        f"/entity-suggestions/{semantic_id}/workflow/apply",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/entity-suggestions/{semantic_id}/workflow?profile_id={profile_id}",
            "friendly_name": "",
            "area_id": "",
            "new_area_name": "",
            "device_class": "temperature",
            "labels_csv": "",
        },
        follow_redirects=False,
    )
    assert semantic_apply.status_code == 303

    labels_apply = client.post(
        f"/entity-suggestions/{labels_id}/workflow/apply",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/entity-suggestions/{labels_id}/workflow?profile_id={profile_id}",
            "friendly_name": "",
            "area_id": "",
            "new_area_name": "",
            "device_class": "",
            "labels_csv": "label_custom",
        },
        follow_redirects=False,
    )
    assert labels_apply.status_code == 303

    semantic_update = next(item for item in updates if item["entity_id"] == "sensor.semantic_missing")
    labels_update = next(item for item in updates if item["entity_id"] == "sensor.labels_missing")
    assert semantic_update["device_class"] == "temperature"
    assert labels_update["labels"] == ["label_custom"]


def test_workflow_apply_rejects_manual_only_and_skip_sets_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-manual-skip")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.manual_only",
                "state": "55",
                "attributes": {
                    "friendly_name": "Manual Only",
                    "device_class": "temperature",
                    "state_class": "measurement",
                    "area_id": "office_area",
                },
            },
            {
                "entity_id": "sensor.skip_me",
                "state": "65",
                "attributes": {
                    "friendly_name": "Skip Me",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            },
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [{"area_id": "office_area", "name": "Office"}],
            "devices": [],
            "entities": [
                {"entity_id": "sensor.manual_only", "labels": ["label_existing"]},
            ],
            "labels": [{"label_id": "label_existing", "name": "Existing"}],
            "floors": [],
        }

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    run_sync_and_suggestions(client, profile_id, csrf_token)
    suggestions_payload = client.get(f"/api/entity-suggestions?profile_id={profile_id}").json()
    by_entity = {item["entity_id"]: item for item in suggestions_payload["items"]}

    manual_id = by_entity["sensor.manual_only"]["id"]
    skip_id = by_entity["sensor.skip_me"]["id"]

    manual_apply = client.post(
        f"/entity-suggestions/{manual_id}/workflow/apply",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/entity-suggestions/{manual_id}/workflow?profile_id={profile_id}",
            "friendly_name": "Manual Only Updated",
            "area_id": "",
            "new_area_name": "",
            "device_class": "",
            "labels_csv": "",
        },
        follow_redirects=False,
    )
    assert manual_apply.status_code == 303

    manual_detail = client.get(f"/api/entity-suggestions/{manual_id}?profile_id={profile_id}").json()
    assert manual_detail["workflow_status"] == "error"
    assert "no workflow-editable issues" in manual_detail["workflow_error"].lower()

    skip_response = client.post(
        f"/entity-suggestions/{skip_id}/workflow/skip",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/entity-suggestions/workflow?profile_id={profile_id}",
            "reason": "Will handle later",
        },
        follow_redirects=False,
    )
    assert skip_response.status_code == 303

    skip_detail = client.get(f"/api/entity-suggestions/{skip_id}?profile_id={profile_id}").json()
    assert skip_detail["workflow_status"] == "skipped"


def test_entity_suggestion_recheck_endpoint_and_api_workflow_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="workflow-recheck")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.recheck_target",
                "state": "72",
                "attributes": {
                    "friendly_name": "Recheck Target",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [{"area_id": "office_area", "name": "Office"}],
            "devices": [
                {
                    "id": "dev_recheck",
                    "name_by_user": "Recheck Device",
                    "area_id": "office_area",
                    "labels": ["label_climate"],
                }
            ],
            "entities": [
                {
                    "entity_id": "sensor.recheck_target",
                    "device_id": "dev_recheck",
                    "labels": ["label_climate"],
                }
            ],
            "labels": [{"label_id": "label_climate", "name": "Climate"}],
            "floors": [],
        }

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setenv("HEV_LLM_ENABLED", "false")

    recheck_response = client.post(
        f"/profiles/{profile_id}/entity-suggestions/recheck",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entity-suggestions/workflow?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert recheck_response.status_code == 303
    location = recheck_response.headers.get("location", "")
    assert "/entity-suggestions/workflow" in location
    run_match = re.search(r"suggestion_run_id=(\d+)", location)
    assert run_match is not None
    run_id = int(run_match.group(1))

    list_payload = client.get(
        f"/api/entity-suggestions?profile_id={profile_id}&suggestion_run_id={run_id}"
    ).json()
    assert list_payload["suggestion_run"]["id"] == run_id
    assert list_payload["items"]
    first_item = list_payload["items"][0]
    assert "workflow_status" in first_item
    assert "workflow_error" in first_item
    assert "workflow_updated_at" in first_item
    assert first_item["workflow_status"] == "open"

    detail_payload = client.get(
        f"/api/entity-suggestions/{first_item['id']}?profile_id={profile_id}"
    ).json()
    assert detail_payload["workflow_status"] == "open"
    assert "workflow_error" in detail_payload
    assert "workflow_updated_at" in detail_payload


def test_generate_automation_drafts_and_review_flow(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    csrf_token = extract_csrf(settings_response.text)
    profile_id = create_profile(client, csrf_token, name="draft-home")

    async def fake_fetch_states(_: Any) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": "sensor.living_temp",
                "state": "72",
                "attributes": {
                    "friendly_name": "Living Room Temperature",
                    "device_class": "temperature",
                    "state_class": "measurement",
                },
                "last_changed": "2026-02-15T01:00:00+00:00",
                "last_updated": "2026-02-15T01:00:00+00:00",
            }
        ]

    async def fake_fetch_registry_metadata(_: Any) -> dict[str, list[dict[str, Any]]]:
        return {
            "areas": [{"area_id": "living_room", "name": "Living Room"}],
            "devices": [
                {
                    "id": "dev_temp",
                    "name_by_user": "Living Temp Device",
                    "area_id": "living_room",
                    "labels": ["label_climate"],
                }
            ],
            "entities": [
                {
                    "entity_id": "sensor.living_temp",
                    "device_id": "dev_temp",
                    "labels": ["label_climate"],
                }
            ],
            "labels": [{"label_id": "label_climate", "name": "Climate"}],
            "floors": [],
        }

    async def fake_generate_automation_draft(
        _: Any,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        assert payload["template_id"] == "temperature_alert"
        return {
            "title": "Living Room Temperature Alert",
            "alias": "Living Room Temperature Alert",
            "description": "Notify when temperature is high.",
            "trigger": [{"platform": "numeric_state", "entity_id": "sensor.living_temp", "above": 80}],
            "condition": [],
            "action": [{"service": "notify.notify", "data": {"message": "Temperature alert"}}],
            "rationale": "Temperature sensor should trigger alert automation.",
        }

    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)
    monkeypatch.setattr("app.main.HAClient.fetch_registry_metadata", fake_fetch_registry_metadata)
    monkeypatch.setattr(
        "app.main.OpenAICompatibleLLMClient.generate_automation_draft",
        fake_generate_automation_draft,
    )
    monkeypatch.setenv("HEV_LLM_ENABLED", "true")
    monkeypatch.setenv("HEV_LLM_BASE_URL", "http://llm.local")
    monkeypatch.setenv("HEV_LLM_API_KEY", "token")
    monkeypatch.setenv("HEV_LLM_MODEL", "test-model")

    sync_response = client.post(
        f"/profiles/{profile_id}/sync",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entities?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert sync_response.status_code == 303

    suggestion_response = client.post(
        f"/profiles/{profile_id}/run-entity-suggestions",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/entity-suggestions?profile_id={profile_id}",
        },
        follow_redirects=False,
    )
    assert suggestion_response.status_code == 303

    suggestions_api = client.get(f"/api/entity-suggestions?profile_id={profile_id}")
    assert suggestions_api.status_code == 200
    suggestions_payload = suggestions_api.json()
    suggestion_run_id = suggestions_payload["suggestion_run"]["id"]
    assert suggestions_payload["items"][0]["readiness_status"] == "ready"

    draft_response = client.post(
        f"/profiles/{profile_id}/generate-automation-drafts",
        data={
            "csrf_token": csrf_token,
            "next_url": f"/automation-drafts?profile_id={profile_id}",
            "suggestion_run_id": str(suggestion_run_id),
            "readiness_status": "ready",
        },
        follow_redirects=False,
    )
    assert draft_response.status_code == 303

    drafts_page = client.get(f"/automation-drafts?profile_id={profile_id}")
    assert drafts_page.status_code == 200
    assert "temperature_alert" in drafts_page.text

    drafts_api = client.get(f"/api/automation-drafts?profile_id={profile_id}")
    assert drafts_api.status_code == 200
    drafts_payload = drafts_api.json()
    assert drafts_payload["total"] == 1
    draft_id = drafts_payload["items"][0]["id"]
    assert drafts_payload["items"][0]["generation_status"] == "success"

    accept_response = client.post(
        f"/automation-drafts/{draft_id}/accept",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/automation-drafts?profile_id={profile_id}",
            "review_note": "Looks good",
        },
        follow_redirects=False,
    )
    assert accept_response.status_code == 303

    accepted_detail = client.get(f"/api/automation-drafts/{draft_id}?profile_id={profile_id}")
    assert accepted_detail.status_code == 200
    assert accepted_detail.json()["review_status"] == "accepted"

    reject_response = client.post(
        f"/automation-drafts/{draft_id}/reject",
        data={
            "csrf_token": csrf_token,
            "profile_id": str(profile_id),
            "next_url": f"/automation-drafts?profile_id={profile_id}",
            "review_note": "Needs changes",
        },
        follow_redirects=False,
    )
    assert reject_response.status_code == 303

    rejected_detail = client.get(f"/api/automation-drafts/{draft_id}?profile_id={profile_id}")
    assert rejected_detail.status_code == 200
    assert rejected_detail.json()["review_status"] == "rejected"
