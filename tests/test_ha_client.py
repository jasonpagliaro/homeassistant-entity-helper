from __future__ import annotations

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
