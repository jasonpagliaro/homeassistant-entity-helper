from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class SuggestionWorker:
    def __init__(self, processor: Callable[[int], Awaitable[None]]) -> None:
        self._processor = processor
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._pending: set[int] = set()
        self._task: asyncio.Task[None] | None = None

    def enqueue(self, run_id: int) -> None:
        if run_id in self._pending:
            return
        self._pending.add(run_id)
        self._queue.put_nowait(run_id)

    def recover(self, run_ids: list[int]) -> None:
        for run_id in run_ids:
            self.enqueue(run_id)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run_loop(self) -> None:
        while True:
            run_id = await self._queue.get()
            try:
                await self._processor(run_id)
            finally:
                self._pending.discard(run_id)
