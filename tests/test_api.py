from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import db
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


def test_settings_sync_and_export_flow(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    response = client.get("/settings")
    assert response.status_code == 200

    csrf_token = extract_csrf(response.text)
    profile_id = extract_profile_id(response.text)

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

    monkeypatch.setattr("app.main.HAClient.test_connection", fake_test_connection)
    monkeypatch.setattr("app.main.HAClient.fetch_states", fake_fetch_states)

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

    detail_response = client.get(f"/entities/light.kitchen?profile_id={profile_id}")
    assert detail_response.status_code == 200
    assert "Kitchen Light" in detail_response.text

    export_json_response = client.get(f"/export/json?profile_id={profile_id}&q=light")
    assert export_json_response.status_code == 200
    exported = json.loads(export_json_response.text)
    assert len(exported) == 1
    assert exported[0]["entity_id"] == "light.kitchen"
    assert "pulled_at" in exported[0]

    export_csv_response = client.get(f"/export/csv?profile_id={profile_id}&domain=light")
    assert export_csv_response.status_code == 200
    assert "text/csv" in export_csv_response.headers["content-type"]
    assert "light.kitchen" in export_csv_response.text
