from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.verify_tls = verify_tls
        self.timeout_seconds = timeout_seconds
        self.transport = transport

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

    async def fetch_registry_metadata(self) -> dict[str, list[dict[str, Any]]]:
        try:
            import websockets
            from websockets.exceptions import WebSocketException
        except ImportError as exc:
            raise HAClientError(
                "Missing dependency 'websockets'; unable to pull registry metadata."
            ) from exc

        ws_url = self._websocket_url()
        ssl_context = None
        if ws_url.startswith("wss://") and not self.verify_tls:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        metadata: dict[str, list[dict[str, Any]]] = {
            key: [] for key in self._REGISTRY_COMMANDS.keys()
        }

        try:
            async with websockets.connect(
                ws_url,
                open_timeout=self.timeout_seconds,
                close_timeout=self.timeout_seconds,
                ssl=ssl_context,
                max_size=2_000_000,
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
                for bucket, command_type in self._REGISTRY_COMMANDS.items():
                    await asyncio.wait_for(
                        websocket.send(
                            json.dumps({"id": message_id, "type": command_type})
                        ),
                        timeout=self.timeout_seconds,
                    )
                    result = await asyncio.wait_for(
                        self._wait_for_ws_result(websocket, message_id),
                        timeout=self.timeout_seconds,
                    )
                    message_id += 1

                    if result.get("type") != "result":
                        raise HAClientError(
                            f"Unexpected response format for WebSocket command '{command_type}'."
                        )

                    if not result.get("success"):
                        error = result.get("error")
                        if isinstance(error, dict):
                            code = str(error.get("code", ""))
                            message = str(error.get("message", "unknown error"))
                        else:
                            code = ""
                            message = "unknown error"

                        # Keep sync resilient across HA versions where some registries are unavailable.
                        if code in {"unknown_command", "not_found"}:
                            metadata[bucket] = []
                            continue

                        raise HAClientError(
                            f"Registry command '{command_type}' failed: {message}"
                        )

                    payload = result.get("result")
                    if not isinstance(payload, list):
                        metadata[bucket] = []
                        continue
                    metadata[bucket] = [item for item in payload if isinstance(item, dict)]

        except HAClientError:
            raise
        except (WebSocketException, OSError, TimeoutError) as exc:
            raise HAClientError(
                f"Unable to reach Home Assistant WebSocket API: {exc}"
            ) from exc

        return metadata
