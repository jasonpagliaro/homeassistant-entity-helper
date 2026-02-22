from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.llm_client import (
    LLMClientError,
    LLMSettings,
    OpenAICompatibleLLMClient,
    _validate_automation_draft_response,
    _validate_entity_suggestion_response,
)


def test_validate_entity_suggestion_response() -> None:
    payload = {
        "proposed_updates": [
            {"field": "area_name", "value": "Kitchen", "confidence": 1.4, "reason": "match"},
            {"field": "friendly_name", "value": "Kitchen Light", "confidence": 0.8, "reason": "name"},
        ],
        "type_hints": {"device_class": "light"},
        "notes": "ok",
    }
    normalized = _validate_entity_suggestion_response(payload)
    assert len(normalized["proposed_updates"]) == 2
    assert normalized["proposed_updates"][0]["confidence"] == 1.0
    assert normalized["type_hints"]["device_class"] == "light"


def test_validate_automation_draft_response_missing_trigger() -> None:
    with pytest.raises(LLMClientError):
        _validate_automation_draft_response(
            {
                "title": "Draft",
                "alias": "Draft",
                "description": "",
                "trigger": [],
                "condition": [],
                "action": [{"action": "light.turn_on"}],
                "rationale": "",
            }
        )


@pytest.mark.asyncio
async def test_llm_request_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            request = httpx.Request("POST", "https://example.invalid")
            raise httpx.RequestError("network down", request=request)

    monkeypatch.setattr("app.llm_client.httpx.AsyncClient", lambda *args, **kwargs: FakeClient())

    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="https://example.invalid",
            api_key="token",
            model="test-model",
            timeout_seconds=1,
            max_concurrency=1,
        )
    )
    with pytest.raises(LLMClientError, match="Unable to reach LLM provider"):
        await client.suggest_entity_metadata({"entity": {"entity_id": "sensor.temp"}})


@pytest.mark.asyncio
async def test_llm_invalid_json_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "title": "Draft",
                                    "alias": "Draft",
                                    "description": "",
                                    "trigger": [{"platform": "state"}],
                                    "condition": [],
                                    "action": [{"action": "light.turn_on"}],
                                    "rationale": "ok",
                                }
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            return FakeResponse()

    monkeypatch.setattr("app.llm_client.httpx.AsyncClient", lambda *args, **kwargs: FakeClient())

    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="https://example.invalid",
            api_key="token",
            model="test-model",
            timeout_seconds=1,
            max_concurrency=1,
        )
    )
    response = await client.generate_automation_draft({"entity": {"entity_id": "sensor.temp"}})
    assert response["alias"] == "Draft"


@pytest.mark.asyncio
async def test_chat_json_requires_key_for_remote_when_not_allowed() -> None:
    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="https://api.example.com/v1",
            api_key="",
            model="test-model",
            timeout_seconds=1,
            max_concurrency=1,
            allow_missing_api_key=False,
        )
    )
    with pytest.raises(LLMClientError, match="not enabled or configured"):
        await client.chat_json("system", {"ping": "pong"})


@pytest.mark.asyncio
async def test_chat_json_allows_local_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"ok": True}),
                        }
                    }
                ]
            }

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            headers = kwargs.get("headers") or {}
            assert "Authorization" not in headers
            return FakeResponse()

    monkeypatch.setattr("app.llm_client.httpx.AsyncClient", lambda *args, **kwargs: FakeClient())

    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="http://localhost:11434/v1",
            api_key="",
            model="llama3.1",
            timeout_seconds=1,
            max_concurrency=1,
            allow_missing_api_key=True,
        )
    )
    response = await client.chat_json("system", {"ping": "pong"})
    assert response["ok"] is True
