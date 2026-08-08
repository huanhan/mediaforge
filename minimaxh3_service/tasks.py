from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .backend import BackendError, BackendSubmission, BackendTask, OutputArtifact
from .models import GenerationRequest
from .workflow import GenerationPlan, build_plan


logger = logging.getLogger(__name__)


class GenerationCanceled(RuntimeError):
    pass


class H3Runner(Protocol):
    def readiness(self) -> dict: ...

    def generate(
        self,
        plan: GenerationPlan,
        output: OutputArtifact,
        cancel: threading.Event,
        progress: Callable[[float], None],
    ) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _TaskRecord:
    task: BackendTask
    plan: GenerationPlan
    output: OutputArtifact
    cancel: threading.Event = field(default_factory=threading.Event)


class LocalTaskManager:
    def __init__(
        self,
        runner: H3Runner,
        output_factory: Callable[[str, str | None], OutputArtifact],
        *,
        max_queue_size: int,
        allow_interrupt_running: bool,
    ) -> None:
        self.runner = runner
        self.output_factory = output_factory
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self.allow_interrupt_running = allow_interrupt_running
        self.records: dict[str, _TaskRecord] = {}
        self._worker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._worker = asyncio.create_task(self._run(), name="minimax-h3-gpu-worker")

    async def close(self) -> None:
        for record in self.records.values():
            if record.task.status in {"queued", "running"}:
                record.cancel.set()
        await self.queue.join()
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        await asyncio.to_thread(self.runner.close)

    async def submit(self, request: GenerationRequest) -> BackendSubmission:
        if self.queue.full():
            raise BackendError("generation queue is full", status_code=429, code="queue_full")
        plan = build_plan(request)
        task_id = uuid.uuid4().hex
        output = self.output_factory(task_id, request.filename_prefix)
        queue_number = self.queue.qsize() + 1
        self.records[task_id] = _TaskRecord(
            task=BackendTask(id=task_id, status="queued", progress=0.0),
            plan=plan,
            output=output,
        )
        await self.queue.put(task_id)
        return BackendSubmission(
            id=task_id,
            queue_number=queue_number,
            frame_count=plan.frame_count,
            effective_duration_seconds=plan.effective_duration_seconds,
        )

    def get(self, task_id: str) -> BackendTask | None:
        record = self.records.get(task_id)
        if record is None:
            return None
        task = record.task
        return BackendTask(
            id=task.id,
            status=task.status,
            progress=task.progress,
            outputs=list(task.outputs),
            error=task.error,
            raw_status=dict(task.raw_status) if task.raw_status else None,
        )

    def cancel(self, task_id: str) -> BackendTask | None:
        record = self.records.get(task_id)
        if record is None:
            return None
        if record.task.status == "queued":
            record.cancel.set()
            record.task.status = "canceled"
        elif record.task.status == "running":
            if not self.allow_interrupt_running:
                raise BackendError(
                    "task is already running; enable H3_ALLOW_INTERRUPT_RUNNING to cancel between sampling steps",
                    status_code=409,
                    code="task_running",
                )
            record.cancel.set()
        return self.get(task_id)

    async def _run(self) -> None:
        while True:
            task_id = await self.queue.get()
            record = self.records[task_id]
            try:
                if record.cancel.is_set():
                    continue
                record.task.status = "running"
                record.task.progress = 0.01

                def report(value: float) -> None:
                    if self._loop is not None:
                        self._loop.call_soon_threadsafe(self._set_progress, task_id, value)

                await asyncio.to_thread(
                    self.runner.generate,
                    record.plan,
                    record.output,
                    record.cancel,
                    report,
                )
                if record.cancel.is_set():
                    raise GenerationCanceled("generation canceled")
                record.task.status = "succeeded"
                record.task.progress = 1.0
                record.task.outputs = [record.output]
            except GenerationCanceled:
                record.output.path.unlink(missing_ok=True)
                record.task.status = "canceled"
                record.task.progress = None
            except Exception as exc:
                logger.exception("H3 generation %s failed", task_id)
                record.output.path.unlink(missing_ok=True)
                record.task.status = "failed"
                record.task.progress = None
                record.task.error = str(exc)
            finally:
                self.queue.task_done()

    def _set_progress(self, task_id: str, value: float) -> None:
        record = self.records.get(task_id)
        if record is not None and record.task.status == "running":
            record.task.progress = min(0.99, max(0.0, value))
