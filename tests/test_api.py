from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import db
from app.ha_client import HAClientError
from app.main import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("HEV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("APP_NAME", "HA Entity Vault Test")

    db.reset_engine_for_tests()
    app = create_app()

    with TestClient(app) as test_client:
        yield test_client

    db.reset_engine_for_tests()


def extract_csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def extract_profile_id(html: str) -> int:
    match = re.search(r"/profiles/(\d+)/update", html)
    assert match is not None
    return int(match.group(1))


def update_profile(client: TestClient, profile_id: int, csrf_token: str) -> None:
    update_response = client.post(
        f"/profiles/{profile_id}/update",
        data={
            "csrf_token": csrf_token,
            "name": "default",
            "base_url": "http://ha.local:8123",
            "token": "test-token",
            "token_env_var": "HA_TOKEN",
            "verify_tls": "on",
            "timeout_seconds": "10",
        },
        follow_redirects=False,
    )
    assert update_response.status_code == 303


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


def test_sync_modal_markup_and_form_attributes(client: TestClient) -> None:
    settings_response = client.get("/settings")
    assert settings_response.status_code == 200
    settings_html = settings_response.text
    assert_sync_modal_markup(settings_html)
    assert_form_has_sync_modal_attrs(
        settings_html,
        r"/profiles/\d+/sync",
        "Syncing entities...",
    )
    assert_form_has_sync_modal_attrs(
        settings_html,
        r"/profiles/\d+/sync-config",
        "Syncing config items...",
    )
    assert_form_lacks_sync_modal_attrs(settings_html, r"/profiles/\d+/test")
    assert_form_lacks_sync_modal_attrs(settings_html, r"/profiles/\d+/delete")

    entities_response = client.get("/entities")
    assert entities_response.status_code == 200
    entities_html = entities_response.text
    assert_sync_modal_markup(entities_html)
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
    assert_form_has_sync_modal_attrs(
        config_items_html,
        r"/profiles/\d+/sync-config",
        "Syncing config items...",
    )


def test_settings_sync_and_export_flow(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    response = client.get("/settings")
    assert response.status_code == 200

    csrf_token = extract_csrf(response.text)
    profile_id = extract_profile_id(response.text)
    update_profile(client, profile_id, csrf_token)

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
    profile_id = extract_profile_id(response.text)
    update_profile(client, profile_id, csrf_token)

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
    profile_id = extract_profile_id(response.text)
    update_profile(client, profile_id, csrf_token)

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
    profile_id = extract_profile_id(response.text)
    update_profile(client, profile_id, csrf_token)

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
