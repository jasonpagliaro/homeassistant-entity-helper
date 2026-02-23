from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.llm_client import (
    LLMClientError,
    LLMSettings,
    OpenAICompatibleLLMClient,
    _chat_completions_path,
    _models_path,
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


def test_openai_compatible_endpoint_path_normalization() -> None:
    assert _chat_completions_path("https://api.openai.com/v1") == "https://api.openai.com/v1/chat/completions"
    assert _chat_completions_path("https://api.anthropic.com/v1") == "https://api.anthropic.com/v1/chat/completions"
    assert (
        _chat_completions_path("https://generativelanguage.googleapis.com/v1beta/openai")
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert _chat_completions_path("https://api.example.com") == "https://api.example.com/v1/chat/completions"

    assert _models_path("https://api.openai.com/v1") == "https://api.openai.com/v1/models"
    assert _models_path("https://generativelanguage.googleapis.com/v1beta/openai") == (
        "https://generativelanguage.googleapis.com/v1beta/openai/models"
    )
    assert _models_path("https://api.example.com") == "https://api.example.com/v1/models"


@pytest.mark.asyncio
async def test_list_models_parses_multiple_payload_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": [
                    {"id": "gpt-4o"},
                    {"id": "models/gemini-2.5-flash"},
                    {"name": "claude-sonnet-4-5"},
                    {"model": "mixtral-8x7b"},
                    "llama3.1",
                ]
            }

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def get(self, *args: Any, **kwargs: Any) -> Any:
            return FakeResponse()

    monkeypatch.setattr("app.llm_client.httpx.AsyncClient", lambda *args, **kwargs: FakeClient())
    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="https://api.example.com/v1",
            api_key="token",
            model="unused",
            timeout_seconds=1,
            max_concurrency=1,
        )
    )
    models = await client.list_models()
    assert models == [
        "gpt-4o",
        "gemini-2.5-flash",
        "claude-sonnet-4-5",
        "mixtral-8x7b",
        "llama3.1",
    ]


@pytest.mark.asyncio
async def test_chat_json_retries_without_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingResponse:
        def __init__(self) -> None:
            request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
            self.response = httpx.Response(
                400,
                request=request,
                json={"error": {"message": "Unsupported response_format value."}},
            )

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError("bad request", request=self.response.request, response=self.response)

    class SuccessfulResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}

    class FakeClient:
        def __init__(self) -> None:
            self.post_bodies: list[dict[str, Any]] = []

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            body = kwargs.get("json")
            if isinstance(body, dict):
                self.post_bodies.append(body)
            else:
                self.post_bodies.append({})
            if len(self.post_bodies) == 1:
                return FailingResponse()
            return SuccessfulResponse()

    fake_client = FakeClient()
    monkeypatch.setattr("app.llm_client.httpx.AsyncClient", lambda *args, **kwargs: fake_client)

    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="https://api.example.com/v1",
            api_key="token",
            model="test-model",
            timeout_seconds=1,
            max_concurrency=1,
        )
    )
    response = await client.chat_json("system", {"ping": "pong"})
    assert response["ok"] is True
    assert len(fake_client.post_bodies) == 2
    assert "response_format" in fake_client.post_bodies[0]
    assert "response_format" not in fake_client.post_bodies[1]


@pytest.mark.asyncio
async def test_chat_json_retries_with_max_completion_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingResponse:
        def __init__(self) -> None:
            request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
            self.response = httpx.Response(
                400,
                request=request,
                json={
                    "error": {
                        "message": (
                            "Unsupported parameter: 'max_tokens' is not supported with this model. "
                            "Use 'max_completion_tokens' instead."
                        )
                    }
                },
            )

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError("bad request", request=self.response.request, response=self.response)

    class SuccessfulResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}

    class FakeClient:
        def __init__(self) -> None:
            self.post_bodies: list[dict[str, Any]] = []

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            body = kwargs.get("json")
            if isinstance(body, dict):
                self.post_bodies.append(body)
            else:
                self.post_bodies.append({})
            if len(self.post_bodies) == 1:
                return FailingResponse()
            return SuccessfulResponse()

    fake_client = FakeClient()
    monkeypatch.setattr("app.llm_client.httpx.AsyncClient", lambda *args, **kwargs: fake_client)

    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="https://api.example.com/v1",
            api_key="token",
            model="test-model",
            timeout_seconds=1,
            max_concurrency=1,
        )
    )
    response = await client.chat_json("system", {"ping": "pong"})
    assert response["ok"] is True
    assert len(fake_client.post_bodies) == 2
    assert "max_tokens" in fake_client.post_bodies[0]
    assert "max_completion_tokens" not in fake_client.post_bodies[0]
    assert "max_tokens" not in fake_client.post_bodies[1]
    assert "max_completion_tokens" in fake_client.post_bodies[1]


@pytest.mark.asyncio
async def test_test_connection_verbose_captures_http_400_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingResponse:
        def __init__(self) -> None:
            request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
            self.response = httpx.Response(
                400,
                request=request,
                json={"error": {"message": "Bad request payload"}},
            )

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError("bad request", request=self.response.request, response=self.response)

        def json(self) -> dict[str, Any]:
            return {"error": {"message": "Bad request payload"}}

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            return FailingResponse()

    monkeypatch.setattr("app.llm_client.httpx.AsyncClient", lambda *args, **kwargs: FakeClient())
    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="https://api.example.com/v1",
            api_key="token",
            model="test-model",
            timeout_seconds=1,
            max_concurrency=1,
        )
    )
    result = await client.test_connection_verbose()
    assert result["ok"] is False
    assert "HTTP 400" in result["message"]
    debug = result["debug"]
    assert debug["endpoint"].endswith("/chat/completions")
    assert debug["headers"]["Authorization"] == "Bearer ***redacted***"
    assert debug["attempts"][0]["response_status"] == 400


@pytest.mark.asyncio
async def test_chat_json_retries_with_higher_output_tokens_when_content_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyContentLengthResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": ""},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 900,
                    "total_tokens": 1000,
                },
            }

    class SuccessResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {"content": json.dumps({"ok": True})},
                    }
                ]
            }

    class FakeClient:
        def __init__(self) -> None:
            self.post_bodies: list[dict[str, Any]] = []

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            body = kwargs.get("json")
            if isinstance(body, dict):
                self.post_bodies.append(body)
            else:
                self.post_bodies.append({})
            if len(self.post_bodies) == 1:
                return EmptyContentLengthResponse()
            return SuccessResponse()

    fake_client = FakeClient()
    monkeypatch.setattr("app.llm_client.httpx.AsyncClient", lambda *args, **kwargs: fake_client)

    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="https://api.openai.com/v1",
            api_key="token",
            model="gpt-5.2",
            timeout_seconds=1,
            max_concurrency=1,
            max_output_tokens=900,
        )
    )
    response = await client.chat_json("system", {"ping": "pong"})
    assert response["ok"] is True
    assert len(fake_client.post_bodies) == 2
    assert int(fake_client.post_bodies[0]["max_tokens"]) == 900
    assert int(fake_client.post_bodies[1]["max_tokens"]) > 900


@pytest.mark.asyncio
async def test_chat_json_request_error_message_includes_type_and_url(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            raise httpx.ReadTimeout("", request=request)

    monkeypatch.setattr("app.llm_client.httpx.AsyncClient", lambda *args, **kwargs: FakeClient())

    client = OpenAICompatibleLLMClient(
        LLMSettings(
            enabled=True,
            base_url="https://api.openai.com/v1",
            api_key="token",
            model="gpt-5.2",
            timeout_seconds=1,
            max_concurrency=1,
        )
    )
    debug_context: dict[str, Any] = {}
    with pytest.raises(LLMClientError) as exc:
        await client.chat_json("system", {"ping": "pong"}, debug_context=debug_context)
    message = str(exc.value)
    assert "ReadTimeout" in message
    assert "request timed out" in message
    assert "https://api.openai.com/v1/chat/completions" in message
    assert debug_context["attempts"][0]["network_error_type"] == "ReadTimeout"
    assert "request timed out" in debug_context["attempts"][0]["network_error"]
