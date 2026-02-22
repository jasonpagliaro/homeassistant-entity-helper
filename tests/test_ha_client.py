from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.ha_client import HAClient, HAClientError


@pytest.mark.asyncio
async def test_test_connection_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/config"
        assert request.headers["Authorization"].startswith("Bearer ")
        return httpx.Response(200, json={"version": "2026.2.1"})

    client = HAClient(
        base_url="http://example.local/",
        token="abc123",
        transport=httpx.MockTransport(handler),
    )

    result = await client.test_connection()
    assert result["version"] == "2026.2.1"
    assert client.base_url == "http://example.local"


@pytest.mark.asyncio
async def test_fetch_states_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/states"
        return httpx.Response(
            200,
            json=[
                {
                    "entity_id": "light.kitchen",
                    "state": "on",
                    "attributes": {"friendly_name": "Kitchen"},
                }
            ],
        )

    client = HAClient(
        base_url="http://example.local",
        token="abc123",
        transport=httpx.MockTransport(handler),
    )

    states = await client.fetch_states()
    assert len(states) == 1
    assert states[0]["entity_id"] == "light.kitchen"


@pytest.mark.asyncio
async def test_fetch_config_rest_endpoints_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/config/automation/config/evening_mode":
            return httpx.Response(200, json={"alias": "Evening Mode"})
        if request.url.path == "/api/config/script/config/goodnight":
            return httpx.Response(200, json={"alias": "Goodnight"})
        if request.url.path == "/api/config/scene/config/movie_time":
            return httpx.Response(200, json={"name": "Movie Time", "entities": {}})
        return httpx.Response(404, json={"message": "Not found"})

    client = HAClient(
        base_url="http://example.local",
        token="abc123",
        transport=httpx.MockTransport(handler),
    )

    automation = await client.fetch_automation_config("evening_mode")
    script = await client.fetch_script_config("goodnight")
    scene = await client.fetch_scene_config("movie_time")

    assert automation["alias"] == "Evening Mode"
    assert script["alias"] == "Goodnight"
    assert scene["name"] == "Movie Time"


@pytest.mark.asyncio
async def test_fetch_automation_config_ws_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ws_request(self: HAClient, command_type: str, **payload: Any) -> Any:
        assert command_type == "automation/config"
        assert payload["entity_id"] == "automation.evening_mode"
        return {"config": {"alias": "Evening Mode"}}

    monkeypatch.setattr(HAClient, "_ws_request", fake_ws_request)

    client = HAClient(base_url="http://example.local", token="abc123")
    automation = await client.fetch_automation_config_ws("automation.evening_mode")

    assert automation["alias"] == "Evening Mode"


@pytest.mark.asyncio
async def test_fetch_script_config_ws_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ws_request(self: HAClient, command_type: str, **payload: Any) -> Any:
        assert command_type == "script/config"
        assert payload["entity_id"] == "script.goodnight"
        return {"config": {"alias": "Goodnight"}}

    monkeypatch.setattr(HAClient, "_ws_request", fake_ws_request)

    client = HAClient(base_url="http://example.local", token="abc123")
    script = await client.fetch_script_config_ws("script.goodnight")

    assert script["alias"] == "Goodnight"


@pytest.mark.asyncio
async def test_auth_error_is_actionable() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Unauthorized"})

    client = HAClient(
        base_url="http://example.local",
        token="bad-token",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(HAClientError) as exc_info:
        await client.test_connection()

    assert exc_info.value.status_code == 401
    assert "Authentication failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_not_found_error_includes_status() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not found"})

    client = HAClient(
        base_url="http://example.local",
        token="abc123",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(HAClientError) as exc_info:
        await client.fetch_scene_config("missing_scene")

    assert exc_info.value.status_code == 404
    assert "HTTP 404" in str(exc_info.value)


def test_websocket_url_normalization() -> None:
    client = HAClient(base_url="https://ha.local:8123/root/", token="abc123")
    assert client._websocket_url() == "wss://ha.local:8123/root/api/websocket"


@pytest.mark.asyncio
async def test_update_entity_registry_entry_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ws_request(self: HAClient, command_type: str, **payload: Any) -> Any:
        assert command_type == "config/entity_registry/update"
        assert payload["entity_id"] == "sensor.office_temp"
        assert payload["name"] == "Office Temperature"
        assert payload["area_id"] == "office"
        assert payload["device_class"] == "temperature"
        assert payload["labels"] == ["label_climate"]
        return {"entity_entry": {"entity_id": payload["entity_id"]}}

    monkeypatch.setattr(HAClient, "_ws_request", fake_ws_request)
    client = HAClient(base_url="http://example.local", token="abc123")
    entry = await client.update_entity_registry_entry(
        entity_id="sensor.office_temp",
        name="Office Temperature",
        area_id="office",
        device_class="temperature",
        labels=["label_climate"],
    )
    assert entry["entity_id"] == "sensor.office_temp"


@pytest.mark.asyncio
async def test_update_entity_registry_entry_requires_changes() -> None:
    client = HAClient(base_url="http://example.local", token="abc123")
    with pytest.raises(HAClientError) as exc_info:
        await client.update_entity_registry_entry(entity_id="sensor.office_temp")
    assert "No entity registry changes were provided" in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_area_registry_entry_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ws_request(self: HAClient, command_type: str, **payload: Any) -> Any:
        assert command_type == "config/area_registry/create"
        assert payload["name"] == "Office"
        return {"area_id": "office", "name": "Office"}

    monkeypatch.setattr(HAClient, "_ws_request", fake_ws_request)
    client = HAClient(base_url="http://example.local", token="abc123")
    area = await client.create_area_registry_entry(name=" Office ")
    assert area["area_id"] == "office"


@pytest.mark.asyncio
async def test_fetch_area_registry_entries_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ws_request(self: HAClient, command_type: str, **payload: Any) -> Any:
        assert command_type == "config/area_registry/list"
        assert payload == {}
        return [
            {"area_id": "office", "name": "Office"},
            "invalid-entry",
            {"id": "garage", "name": "Garage"},
        ]

    monkeypatch.setattr(HAClient, "_ws_request", fake_ws_request)
    client = HAClient(base_url="http://example.local", token="abc123")

    areas = await client.fetch_area_registry_entries()
    assert areas == [
        {"area_id": "office", "name": "Office"},
        {"id": "garage", "name": "Garage"},
    ]


@pytest.mark.asyncio
async def test_fetch_area_registry_entries_rejects_non_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_ws_request(self: HAClient, command_type: str, **payload: Any) -> Any:
        assert command_type == "config/area_registry/list"
        return {"area_id": "office"}

    monkeypatch.setattr(HAClient, "_ws_request", fake_ws_request)
    client = HAClient(base_url="http://example.local", token="abc123")

    with pytest.raises(HAClientError) as exc_info:
        await client.fetch_area_registry_entries()
    assert "Unexpected response format from config/area_registry/list." in str(exc_info.value)
