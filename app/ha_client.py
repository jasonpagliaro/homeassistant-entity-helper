from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

_UNSET = object()


class HAClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HAClient:
    _REGISTRY_COMMANDS: dict[str, str] = {
        "areas": "config/area_registry/list",
        "devices": "config/device_registry/list",
        "entities": "config/entity_registry/list",
        "labels": "config/label_registry/list",
        "floors": "config/floor_registry/list",
    }

    def __init__(
        self,
        base_url: str,
        token: str,
        verify_tls: bool = True,
        timeout_seconds: int = 10,
        transport: httpx.AsyncBaseTransport | None = None,
        max_ws_message_size: int = 8_000_000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.verify_tls = verify_tls
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.max_ws_message_size = max_ws_message_size

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def _get_json(self, path: str) -> Any:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                verify=self.verify_tls,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.get(path)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                message = "Authentication failed. Check your Home Assistant token."
            else:
                message = f"Home Assistant API returned HTTP {status}."
            raise HAClientError(message, status_code=status) from exc
        except httpx.RequestError as exc:
            raise HAClientError(f"Unable to reach Home Assistant: {exc}") from exc

    async def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                verify=self.verify_tls,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                message = "Authentication failed. Check your Home Assistant token."
            else:
                message = f"Home Assistant API returned HTTP {status}."
            raise HAClientError(message, status_code=status) from exc
        except httpx.RequestError as exc:
            raise HAClientError(f"Unable to reach Home Assistant: {exc}") from exc

    def _websocket_url(self) -> str:
        parsed = urlsplit(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        base_path = parsed.path.rstrip("/")
        ws_path = f"{base_path}/api/websocket" if base_path else "/api/websocket"
        return urlunsplit((ws_scheme, parsed.netloc, ws_path, "", ""))

    async def _ws_recv_json(self, websocket: Any) -> dict[str, Any]:
        raw_message = await websocket.recv()
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8", errors="replace")
        if not isinstance(raw_message, str):
            raise HAClientError("Unexpected WebSocket message type from Home Assistant.")

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            raise HAClientError("Invalid JSON payload from Home Assistant WebSocket API.") from exc

        if not isinstance(payload, dict):
            raise HAClientError("Unexpected Home Assistant WebSocket payload format.")
        return payload

    async def _wait_for_ws_result(self, websocket: Any, message_id: int) -> dict[str, Any]:
        while True:
            message = await self._ws_recv_json(websocket)
            if message.get("id") != message_id:
                continue
            return message

    async def test_connection(self) -> dict[str, Any]:
        response = await self._get_json("/api/config")
        if not isinstance(response, dict):
            raise HAClientError("Unexpected response format from /api/config.")
        return response

    async def fetch_states(self) -> list[dict[str, Any]]:
        response = await self._get_json("/api/states")
        if not isinstance(response, list):
            raise HAClientError("Unexpected response format from /api/states.")

        normalized: list[dict[str, Any]] = []
        for item in response:
            if isinstance(item, dict):
                normalized.append(item)
        return normalized

    async def _run_ws_commands(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            import websockets
            from websockets.exceptions import WebSocketException
        except ImportError as exc:
            raise HAClientError(
                "Missing dependency 'websockets'; unable to call Home Assistant WebSocket API."
            ) from exc

        ws_url = self._websocket_url()
        ssl_context = None
        if ws_url.startswith("wss://") and not self.verify_tls:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        results: list[dict[str, Any]] = []
        try:
            async with websockets.connect(
                ws_url,
                open_timeout=self.timeout_seconds,
                close_timeout=self.timeout_seconds,
                ssl=ssl_context,
                max_size=self.max_ws_message_size,
            ) as websocket:
                hello = await asyncio.wait_for(
                    self._ws_recv_json(websocket),
                    timeout=self.timeout_seconds,
                )
                if hello.get("type") != "auth_required":
                    raise HAClientError(
                        "Unexpected Home Assistant WebSocket handshake response."
                    )

                await asyncio.wait_for(
                    websocket.send(json.dumps({"type": "auth", "access_token": self.token})),
                    timeout=self.timeout_seconds,
                )
                auth_result = await asyncio.wait_for(
                    self._ws_recv_json(websocket),
                    timeout=self.timeout_seconds,
                )
                auth_type = auth_result.get("type")
                if auth_type == "auth_invalid":
                    raise HAClientError(
                        "Authentication failed. Check your Home Assistant token."
                    )
                if auth_type != "auth_ok":
                    raise HAClientError(
                        "Unexpected Home Assistant WebSocket authentication response."
                    )

                message_id = 1
                for command in commands:
                    payload = {"id": message_id, **command}
                    await asyncio.wait_for(
                        websocket.send(json.dumps(payload)),
                        timeout=self.timeout_seconds,
                    )
                    result = await asyncio.wait_for(
                        self._wait_for_ws_result(websocket, message_id),
                        timeout=self.timeout_seconds,
                    )
                    if result.get("type") != "result":
                        raise HAClientError(
                            "Unexpected response format from Home Assistant WebSocket API."
                        )
                    results.append(result)
                    message_id += 1
        except HAClientError:
            raise
        except (WebSocketException, OSError, TimeoutError) as exc:
            raise HAClientError(
                f"Unable to reach Home Assistant WebSocket API: {exc}"
            ) from exc

        return results

    @staticmethod
    def _ws_error_details(result: dict[str, Any]) -> tuple[str, str]:
        error = result.get("error")
        if isinstance(error, dict):
            code = str(error.get("code", ""))
            message = str(error.get("message", "unknown error"))
            return code, message
        return "", "unknown error"

    async def _ws_request(self, command_type: str, **payload: Any) -> Any:
        result = (await self._run_ws_commands([{"type": command_type, **payload}]))[0]
        if result.get("success"):
            return result.get("result")

        _, message = self._ws_error_details(result)
        raise HAClientError(f"WebSocket command '{command_type}' failed: {message}")

    async def _fetch_config(self, kind: str, config_key: str) -> dict[str, Any]:
        cleaned_key = config_key.strip()
        if not cleaned_key:
            raise HAClientError("Config key is required.")

        encoded_key = quote(cleaned_key, safe="")
        response = await self._get_json(f"/api/config/{kind}/config/{encoded_key}")
        if not isinstance(response, dict):
            raise HAClientError(
                f"Unexpected response format from /api/config/{kind}/config/{{id}}."
            )
        return response

    async def fetch_automation_config(self, config_key: str) -> dict[str, Any]:
        return await self._fetch_config("automation", config_key)

    async def upsert_automation_config(
        self,
        config_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        cleaned_key = config_key.strip()
        if not cleaned_key:
            raise HAClientError("Config key is required.")
        if not isinstance(payload, dict):
            raise HAClientError("Automation payload must be an object.")

        encoded_key = quote(cleaned_key, safe="")
        response = await self._post_json(f"/api/config/automation/config/{encoded_key}", payload)
        if not isinstance(response, dict):
            raise HAClientError("Unexpected response format from automation upsert endpoint.")
        return response

    async def fetch_script_config(self, config_key: str) -> dict[str, Any]:
        return await self._fetch_config("script", config_key)

    async def fetch_scene_config(self, config_key: str) -> dict[str, Any]:
        return await self._fetch_config("scene", config_key)

    async def fetch_automation_config_ws(self, entity_id: str) -> dict[str, Any]:
        result = await self._ws_request("automation/config", entity_id=entity_id)
        if not isinstance(result, dict):
            raise HAClientError("Unexpected response format from automation/config.")
        config = result.get("config")
        if not isinstance(config, dict):
            raise HAClientError("Unexpected automation config payload format.")
        return config

    async def fetch_script_config_ws(self, entity_id: str) -> dict[str, Any]:
        result = await self._ws_request("script/config", entity_id=entity_id)
        if not isinstance(result, dict):
            raise HAClientError("Unexpected response format from script/config.")
        config = result.get("config")
        if not isinstance(config, dict):
            raise HAClientError("Unexpected script config payload format.")
        return config

    async def fetch_registry_metadata(self) -> dict[str, list[dict[str, Any]]]:
        metadata: dict[str, list[dict[str, Any]]] = {
            key: [] for key in self._REGISTRY_COMMANDS.keys()
        }

        commands = [{"type": command_type} for command_type in self._REGISTRY_COMMANDS.values()]
        results = await self._run_ws_commands(commands)

        for (bucket, command_type), result in zip(
            self._REGISTRY_COMMANDS.items(), results, strict=False
        ):
            if not result.get("success"):
                code, message = self._ws_error_details(result)
                # Keep sync resilient across HA versions where some registries are unavailable.
                if code in {"unknown_command", "not_found"}:
                    metadata[bucket] = []
                    continue
                raise HAClientError(f"Registry command '{command_type}' failed: {message}")

            payload = result.get("result")
            if not isinstance(payload, list):
                metadata[bucket] = []
                continue
            metadata[bucket] = [item for item in payload if isinstance(item, dict)]

        return metadata

    async def fetch_area_registry_entries(self) -> list[dict[str, Any]]:
        try:
            payload = await self._ws_request("config/area_registry/list")
        except HAClientError as exc:
            raise HAClientError(f"Unable to fetch area registry entries: {exc}") from exc

        if not isinstance(payload, list):
            raise HAClientError("Unexpected response format from config/area_registry/list.")
        return [item for item in payload if isinstance(item, dict)]

    async def fetch_device_registry_entries(self) -> list[dict[str, Any]]:
        try:
            payload = await self._ws_request("config/device_registry/list")
        except HAClientError as exc:
            raise HAClientError(f"Unable to fetch device registry entries: {exc}") from exc

        if not isinstance(payload, list):
            raise HAClientError("Unexpected response format from config/device_registry/list.")
        return [item for item in payload if isinstance(item, dict)]

    async def fetch_entity_registry_entries(self) -> list[dict[str, Any]]:
        try:
            payload = await self._ws_request("config/entity_registry/list")
        except HAClientError as exc:
            raise HAClientError(f"Unable to fetch entity registry entries: {exc}") from exc

        if not isinstance(payload, list):
            raise HAClientError("Unexpected response format from config/entity_registry/list.")
        return [item for item in payload if isinstance(item, dict)]

    @staticmethod
    def _clean_entity_id(entity_id: Any) -> str:
        if not isinstance(entity_id, str):
            raise HAClientError("Entity ID must be a string.")
        cleaned = entity_id.strip()
        if "." not in cleaned:
            raise HAClientError("Entity ID must include a domain prefix (domain.object_id).")
        return cleaned

    @staticmethod
    def _clean_optional_str(value: Any, *, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HAClientError(f"{field_name} must be a string or null.")
        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _clean_label_ids(labels: Any) -> list[str]:
        if not isinstance(labels, list):
            raise HAClientError("labels must be an array of strings.")
        cleaned: list[str] = []
        for item in labels:
            if not isinstance(item, str):
                raise HAClientError("labels must only contain strings.")
            normalized = item.strip()
            if normalized:
                cleaned.append(normalized)
        return cleaned

    async def update_entity_registry_entry(
        self,
        *,
        entity_id: str,
        name: Any = _UNSET,
        area_id: Any = _UNSET,
        device_class: Any = _UNSET,
        labels: Any = _UNSET,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "entity_id": self._clean_entity_id(entity_id),
        }

        if name is not _UNSET:
            payload["name"] = self._clean_optional_str(name, field_name="name")
        if area_id is not _UNSET:
            payload["area_id"] = self._clean_optional_str(area_id, field_name="area_id")
        if device_class is not _UNSET:
            payload["device_class"] = self._clean_optional_str(
                device_class,
                field_name="device_class",
            )
        if labels is not _UNSET:
            payload["labels"] = self._clean_label_ids(labels)

        if len(payload) == 1:
            raise HAClientError("No entity registry changes were provided.")

        result = await self._ws_request("config/entity_registry/update", **payload)
        if not isinstance(result, dict):
            raise HAClientError("Unexpected response format from config/entity_registry/update.")
        entry = result.get("entity_entry")
        if not isinstance(entry, dict):
            raise HAClientError("Entity registry update response did not include entity_entry.")
        return entry

    async def create_area_registry_entry(self, *, name: str) -> dict[str, Any]:
        cleaned_name = self._clean_optional_str(name, field_name="name")
        if cleaned_name is None:
            raise HAClientError("Area name is required.")
        result = await self._ws_request("config/area_registry/create", name=cleaned_name)
        if not isinstance(result, dict):
            raise HAClientError("Unexpected response format from config/area_registry/create.")
        if not isinstance(result.get("area_id"), str):
            raise HAClientError("Area creation response did not include area_id.")
        return result
