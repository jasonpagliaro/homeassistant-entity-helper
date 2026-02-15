from __future__ import annotations

from typing import Any

import httpx


class HAClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HAClient:
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
