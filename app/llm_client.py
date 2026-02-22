from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx


class LLMClientError(Exception):
    pass


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LLMSettings:
    enabled: bool
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    max_concurrency: int
    allow_missing_api_key: bool = False
    temperature: float = 0.2
    max_output_tokens: int = 900
    extra_headers: dict[str, str] | None = None


def load_llm_settings() -> LLMSettings:
    enabled = _parse_bool(os.getenv("HEV_LLM_ENABLED"), False)
    base_url = (os.getenv("HEV_LLM_BASE_URL") or "").strip().rstrip("/")
    api_key = (os.getenv("HEV_LLM_API_KEY") or "").strip()
    model = (os.getenv("HEV_LLM_MODEL") or "").strip() or "gpt-4o-mini"
    timeout_seconds = int((os.getenv("HEV_LLM_TIMEOUT_SECONDS") or "20").strip())
    max_concurrency = int((os.getenv("HEV_LLM_MAX_CONCURRENCY") or "4").strip())
    temperature_raw = (os.getenv("HEV_LLM_TEMPERATURE") or "0.2").strip()
    max_output_tokens_raw = (os.getenv("HEV_LLM_MAX_OUTPUT_TOKENS") or "900").strip()
    if timeout_seconds < 1:
        timeout_seconds = 1
    if max_concurrency < 1:
        max_concurrency = 1
    try:
        temperature = float(temperature_raw)
    except ValueError:
        temperature = 0.2
    temperature = max(0.0, min(2.0, temperature))
    try:
        max_output_tokens = int(max_output_tokens_raw)
    except ValueError:
        max_output_tokens = 900
    max_output_tokens = max(1, min(8192, max_output_tokens))
    extra_headers_raw = (os.getenv("HEV_LLM_EXTRA_HEADERS_JSON") or "").strip()
    extra_headers: dict[str, str] | None = None
    if extra_headers_raw:
        try:
            parsed_headers = json.loads(extra_headers_raw)
        except json.JSONDecodeError:
            parsed_headers = {}
        if isinstance(parsed_headers, dict):
            normalized: dict[str, str] = {}
            for key, value in parsed_headers.items():
                if not isinstance(key, str):
                    continue
                cleaned_key = key.strip()
                if not cleaned_key:
                    continue
                normalized[cleaned_key] = str(value)
            if normalized:
                extra_headers = normalized
    return LLMSettings(
        enabled=enabled,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
        allow_missing_api_key=False,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        extra_headers=extra_headers,
    )


def is_probably_local_base_url(base_url: str) -> bool:
    try:
        parsed = urlsplit(base_url)
    except ValueError:
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(".local")


def llm_is_configured(settings: LLMSettings | None = None) -> bool:
    s = settings or load_llm_settings()
    api_key_ok = bool(s.api_key) or s.allow_missing_api_key or is_probably_local_base_url(s.base_url)
    return bool(s.enabled and s.base_url and s.model and api_key_ok)


def _chat_completions_path(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


class OpenAICompatibleLLMClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        if isinstance(self.settings.extra_headers, dict):
            for key, value in self.settings.extra_headers.items():
                cleaned_key = str(key).strip()
                cleaned_value = str(value).strip()
                if cleaned_key and cleaned_value:
                    headers[cleaned_key] = cleaned_value
        return headers

    async def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        return await self._complete_json(system_prompt, user_payload)

    async def test_connection(self) -> dict[str, Any]:
        payload = await self._complete_json(
            "Respond with a small JSON object confirming connectivity.",
            {"ping": "haev"},
        )
        return {"status": "ok", "response_keys": sorted(payload.keys())}

    async def _complete_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not llm_is_configured(self.settings):
            raise LLMClientError("LLM is not enabled or configured.")

        endpoint = _chat_completions_path(self.settings.base_url)
        request_body = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
        }

        headers = self._headers()

        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
                response = await client.post(endpoint, headers=headers, json=request_body)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise LLMClientError(f"LLM provider returned HTTP {status}.") from exc
        except httpx.RequestError as exc:
            raise LLMClientError(f"Unable to reach LLM provider: {exc}") from exc

        payload = response.json()
        if not isinstance(payload, dict):
            raise LLMClientError("Unexpected response format from LLM provider.")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMClientError("LLM provider response missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMClientError("Unexpected choice format from LLM provider.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LLMClientError("LLM response choice missing message.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMClientError("LLM message content is empty.")

        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMClientError("LLM returned invalid JSON content.") from exc

        if not isinstance(decoded, dict):
            raise LLMClientError("LLM JSON response must be an object.")
        return decoded

    async def suggest_entity_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You are a Home Assistant metadata suggestion assistant. "
            "Return JSON only with keys proposed_updates (array), type_hints (object), notes (string). "
            "Each proposed_updates item must include field, value, confidence, reason. "
            "Confidence must be a float in [0,1]. Keep suggestions conservative and actionable."
        )
        response = await self._complete_json(system_prompt, payload)
        return _validate_entity_suggestion_response(response)

    async def generate_automation_draft(self, payload: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You are a Home Assistant automation drafting assistant. "
            "Return JSON only with keys title, alias, description, trigger, condition, action, rationale. "
            "trigger/condition/action must be arrays; condition can be empty."
        )
        response = await self._complete_json(system_prompt, payload)
        return _validate_automation_draft_response(response)


def _validate_entity_suggestion_response(payload: dict[str, Any]) -> dict[str, Any]:
    proposed_updates_raw = payload.get("proposed_updates")
    proposed_updates: list[dict[str, Any]] = []
    if isinstance(proposed_updates_raw, list):
        for item in proposed_updates_raw:
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            value = item.get("value")
            reason = item.get("reason")
            confidence = item.get("confidence")
            if not isinstance(field, str) or not field.strip():
                continue
            if value is None:
                continue
            if not isinstance(reason, str):
                reason = ""
            if isinstance(confidence, (int, float)):
                parsed_conf = max(0.0, min(1.0, float(confidence)))
            else:
                parsed_conf = 0.0
            proposed_updates.append(
                {
                    "field": field.strip(),
                    "value": value,
                    "confidence": parsed_conf,
                    "reason": reason.strip(),
                }
            )

    type_hints = payload.get("type_hints")
    if not isinstance(type_hints, dict):
        type_hints = {}
    notes = payload.get("notes")
    if not isinstance(notes, str):
        notes = ""

    return {
        "proposed_updates": proposed_updates,
        "type_hints": type_hints,
        "notes": notes.strip(),
    }


def _validate_automation_draft_response(payload: dict[str, Any]) -> dict[str, Any]:
    title = payload.get("title")
    alias = payload.get("alias")
    description = payload.get("description")
    trigger = payload.get("trigger")
    condition = payload.get("condition")
    action = payload.get("action")
    rationale = payload.get("rationale")

    if not isinstance(title, str) or not title.strip():
        raise LLMClientError("Draft response missing title.")
    if not isinstance(alias, str) or not alias.strip():
        raise LLMClientError("Draft response missing alias.")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise LLMClientError("Draft response description must be a string.")
    if not isinstance(trigger, list) or not trigger:
        raise LLMClientError("Draft response must include at least one trigger.")
    if not isinstance(condition, list):
        raise LLMClientError("Draft response condition must be an array.")
    if not isinstance(action, list) or not action:
        raise LLMClientError("Draft response must include at least one action.")
    if rationale is None:
        rationale = ""
    if not isinstance(rationale, str):
        raise LLMClientError("Draft response rationale must be a string.")

    return {
        "title": title.strip(),
        "alias": alias.strip(),
        "description": description.strip(),
        "trigger": trigger,
        "condition": condition,
        "action": action,
        "rationale": rationale.strip(),
    }
