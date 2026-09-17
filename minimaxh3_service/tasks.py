from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

from .backend import (
    BackendError,
    BackendSubmission,
    BackendTask,
    OutputArtifact,
    TaskKind,
    UpscaleSubmission,
)
from .models import GenerationRequest, VideoEditRequest, UpscaleRequest, RemovalRequest, SubmittedRemovalTask
from .models import MusicRequest, SubmittedMusicTask
from .models import DepthRequest, SubmittedDepthTask
from .models import SkeletonRequest, SubmittedSkeletonTask
from .workflow import GenerationPlan, SegmentProgress, build_plan


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

    def release_models(self) -> None: ...


class UpscaleRunner(Protocol):
    def readiness(self) -> dict: ...

    def upscale(
        self,
        request: UpscaleRequest,
        output: OutputArtifact,
        cancel: threading.Event,
        progress: Callable[[float], None],
    ) -> None: ...

    def close(self) -> None: ...

    def release_models(self) -> None: ...


@dataclass(slots=True)
class _TaskRecord:
    task: BackendTask
    kind: TaskKind
    payload: GenerationPlan | UpscaleRequest | RemovalRequest | MusicRequest | DepthRequest | SkeletonRequest
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
        idle_release_seconds: float = 5.0,
        upscale_runner: UpscaleRunner | None = None,
        upscale_output_factory: Callable[[str, str | None, str], OutputArtifact] | None = None,
        upscale_idle_release_seconds: float = 5.0,
        removal_runner=None,
        music_runner=None,
        music_output_factory=None,
        depth_runner=None,
        skeleton_runner=None,
    ) -> None:
        self.runner = runner
        self.output_factory = output_factory
        self.upscale_runner = upscale_runner
        self.removal_runner = removal_runner
        self.music_runner = music_runner
        self.depth_runner = depth_runner
        self.skeleton_runner = skeleton_runner
        self.music_output_factory = music_output_factory
        self.upscale_output_factory = upscale_output_factory
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self.allow_interrupt_running = allow_interrupt_running
        self.idle_release_seconds = max(0.0, idle_release_seconds)
        self.upscale_idle_release_seconds = max(0.0, upscale_idle_release_seconds)
        self.records: dict[str, _TaskRecord] = {}
        self._worker: asyncio.Task[None] | None = None
        self._release_task: asyncio.Task[None] | None = None
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
        if self._release_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(self._release_task)
        runners = []
        if self.skeleton_runner is not None:
            runners.append(self.skeleton_runner)
        if self.depth_runner is not None:
            runners.append(self.depth_runner)
        if self.music_runner is not None:
            runners.append(self.music_runner)
        runners.append(self.runner)
        if self.upscale_runner is not None and self.upscale_runner is not self.runner:
            runners.append(self.upscale_runner)
        if self.removal_runner is not None:
            runners.append(self.removal_runner)
        for task_runner in runners:
            await asyncio.to_thread(task_runner.close)

    def _queue_number(self) -> int:
        return self.queue.qsize() + 1

    def _ensure_queue_space(self) -> None:
        if self.queue.full():
            raise BackendError("GPU task queue is full", status_code=429, code="queue_full")

    async def submit(self, request: GenerationRequest | VideoEditRequest) -> BackendSubmission:
        self._ensure_queue_space()
        plan = build_plan(request)
        task_id = uuid.uuid4().hex
        output = self.output_factory(task_id, request.filename_prefix)
        queue_number = self._queue_number()
        self.records[task_id] = _TaskRecord(
            task=BackendTask(
                id=task_id,
                status="queued",
                kind="generation",
                progress=0.0,
                raw_status={"segment_count": len(plan.segments) or 1,
                            "actual_overlap_frames": plan.overlap_frames,
                            "frame_count": plan.frame_count},
            ),
            kind="generation",
            payload=plan,
            output=output,
        )
        await self.queue.put(task_id)
        return BackendSubmission(
            id=task_id,
            queue_number=queue_number,
            frame_count=plan.frame_count,
            effective_duration_seconds=plan.effective_duration_seconds,
            segment_count=len(plan.segments) or 1,
            actual_overlap_frames=plan.overlap_frames,
        )

    async def submit_upscale(self, request: UpscaleRequest) -> UpscaleSubmission:
        if self.upscale_runner is None or self.upscale_output_factory is None:
            raise BackendError(
                "upscaling is not configured",
                status_code=503,
                code="upscale_not_configured",
            )
        self._ensure_queue_space()
        task_id = uuid.uuid4().hex
        output = self.upscale_output_factory(task_id, request.filename_prefix, request.file_id)
        queue_number = self._queue_number()
        self.records[task_id] = _TaskRecord(
            task=BackendTask(
                id=task_id,
                status="queued",
                kind="upscale",
                progress=0.0,
            ),
            kind="upscale",
            payload=request,
            output=output,
        )
        await self.queue.put(task_id)
        return UpscaleSubmission(
            id=task_id,
            queue_number=queue_number,
            width=request.width,
            height=request.height,
        )

    async def submit_removal(self, request: RemovalRequest) -> SubmittedRemovalTask:
        if self.removal_runner is None:
            raise BackendError("removal is not configured", status_code=503, code="removal_not_ready")
        self._ensure_queue_space()
        task_id = uuid.uuid4().hex
        number = self._queue_number()
        self.records[task_id] = _TaskRecord(
            task=BackendTask(id=task_id, status="queued", kind="removal", progress=0.0),
            kind="removal", payload=request,
            output=self.output_factory(task_id, "removal/video"),
        )
        await self.queue.put(task_id)
        return SubmittedRemovalTask(id=task_id, queue_number=number)

    async def submit_music(self, request: MusicRequest) -> SubmittedMusicTask:
        if self.music_runner is None or self.music_output_factory is None:
            raise BackendError("Music generation is not configured", status_code=503, code="music_not_ready")
        self._ensure_queue_space()
        task_id = uuid.uuid4().hex
        number = self._queue_number()
        self.records[task_id] = _TaskRecord(
            task=BackendTask(id=task_id, status="queued", kind="music", progress=0.0),
            kind="music", payload=request, output=self.music_output_factory(task_id),
        )
        await self.queue.put(task_id)
        return SubmittedMusicTask(id=task_id, queue_number=number)

    async def submit_depth(self, request: DepthRequest) -> SubmittedDepthTask:
        if self.depth_runner is None:
            raise BackendError("depth is not configured", status_code=503, code="depth_not_ready")
        self._ensure_queue_space()
        task_id = uuid.uuid4().hex
        number = self._queue_number()
        self.records[task_id] = _TaskRecord(
            task=BackendTask(id=task_id, status="queued", kind="depth", progress=0.0),
            kind="depth", payload=request, output=self.output_factory(task_id, "depth/video"),
        )
        await self.queue.put(task_id)
        return SubmittedDepthTask(id=task_id, queue_number=number)

    async def submit_skeleton(self, request: SkeletonRequest) -> SubmittedSkeletonTask:
        if self.skeleton_runner is None:
            raise BackendError("skeleton extraction is not configured", status_code=503, code="skeleton_not_ready")
        self._ensure_queue_space()
        task_id = uuid.uuid4().hex
        number = self._queue_number()
        self.records[task_id] = _TaskRecord(
            task=BackendTask(id=task_id, status="queued", kind="skeleton", progress=0.0,
                             raw_status={"output_roles": ["skeleton"], "backend": "dwpose"}),
            kind="skeleton", payload=request, output=self.output_factory(task_id, "pose/skeleton"),
        )
        await self.queue.put(task_id)
        return SubmittedSkeletonTask(id=task_id, queue_number=number)

    def get(self, task_id: str) -> BackendTask | None:
        record = self.records.get(task_id)
        if record is None:
            return None
        task = record.task
        return BackendTask(
            id=task.id,
            status=task.status,
            kind=task.kind,
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
            if record.kind == "music" and not getattr(self.music_runner, "supports_interrupt", False):
                raise BackendError("Running music tasks cannot be interrupted by the upstream service; queued tasks can be canceled",
                                   status_code=409, code="task_running")
            if not self.allow_interrupt_running:
                raise BackendError(
                    "task is already running; enable H3_ALLOW_INTERRUPT_RUNNING to cancel it",
                    status_code=409,
                    code="task_running",
                )
            record.cancel.set()
        return self.get(task_id)

    async def _next_task(self, resident_kind: TaskKind | None) -> str | None:
        if resident_kind is None:
            return await self.queue.get()
        timeout = (
            self.idle_release_seconds
            if resident_kind == "generation"
            else self.upscale_idle_release_seconds
        )
        if timeout <= 0:
            try:
                return self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return None
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def _runner_for(self, kind: TaskKind) -> H3Runner | UpscaleRunner:
        if kind == "skeleton":
            return self.skeleton_runner
        if kind == "depth":
            return self.depth_runner
        if kind == "music":
            return self.music_runner
        if kind == "removal":
            if self.removal_runner is None:
                raise RuntimeError("removal is not configured")
            return self.removal_runner
        if kind == "generation":
            return self.runner
        if self.upscale_runner is None:
            raise RuntimeError("upscaling is not configured")
        return self.upscale_runner

    async def _release(self, kind: TaskKind) -> bool:
        task_runner = self._runner_for(kind)
        release_task = asyncio.create_task(
            asyncio.to_thread(task_runner.release_models),
            name=f"minimax-h3-{kind}-idle-release",
        )
        self._release_task = release_task
        release_task.add_done_callback(self._clear_release_task)
        try:
            await asyncio.shield(release_task)
        except Exception:
            logger.exception("failed to release idle %s models", kind)
            return False
        return True

    async def _execute(self, record: _TaskRecord, progress: Callable[[float], None]) -> None:
        if record.kind == "skeleton":
            record.task.outputs = await asyncio.to_thread(
                self.skeleton_runner.process, record.payload, record.output, record.cancel, progress)
            return
        if record.kind == "depth":
            record.task.outputs = await asyncio.to_thread(
                self.depth_runner.process, record.payload, record.output, record.cancel, progress)
            return
        if record.kind == "music":
            await asyncio.to_thread(self.music_runner.generate, record.payload, record.output, record.cancel, progress)
            return
        if record.kind == "removal":
            record.task.outputs = await asyncio.to_thread(
                self.removal_runner.process, record.payload, record.output, record.cancel, progress)
            return
        if record.kind == "generation":
            if not isinstance(record.payload, GenerationPlan):
                raise TypeError("generation task has an invalid payload")
            await asyncio.to_thread(
                self.runner.generate,
                record.payload,
                record.output,
                record.cancel,
                progress,
            )
            return
        if self.upscale_runner is None or not isinstance(record.payload, UpscaleRequest):
            raise TypeError("upscale task has an invalid payload")
        await asyncio.to_thread(
            self.upscale_runner.upscale,
            record.payload,
            record.output,
            record.cancel,
            progress,
        )

    async def _run(self) -> None:
        resident_kind: TaskKind | None = None
        while True:
            task_id = await self._next_task(resident_kind)
            if task_id is None:
                if resident_kind is not None:
                    if await self._release(resident_kind):
                        resident_kind = None
                    else:
                        await asyncio.sleep(1.0)
                continue

            record = self.records[task_id]
            attempted = False
            try:
                if record.cancel.is_set():
                    continue
                if resident_kind is not None and resident_kind != record.kind:
                    released_kind = resident_kind
                    if not await self._release(released_kind):
                        raise RuntimeError(
                            f"cannot release {released_kind} resources before {record.kind} task"
                        )
                    resident_kind = None
                record.task.status = "running"
                record.task.progress = 0.01

                def report(value: float) -> None:
                    if self._loop is not None:
                        self._loop.call_soon_threadsafe(self._set_progress, task_id, value)

                attempted = True
                await self._execute(record, report)
                if record.cancel.is_set():
                    raise GenerationCanceled("task canceled")
                record.task.status = "succeeded"
                record.task.progress = 1.0
                if record.kind not in {"removal", "depth", "skeleton"}:
                    record.task.outputs = [record.output]
            except GenerationCanceled:
                record.output.path.unlink(missing_ok=True)
                for artifact in record.task.outputs:
                    artifact.path.unlink(missing_ok=True)
                record.task.outputs = []
                record.task.status = "canceled"
                record.task.progress = None
            except Exception as exc:
                logger.exception("%s task %s failed", record.kind, task_id)
                record.output.path.unlink(missing_ok=True)
                record.task.status = "failed"
                record.task.progress = None
                record.task.error = str(exc)
            finally:
                self.queue.task_done()
                if attempted:
                    resident_kind = record.kind

    def _clear_release_task(self, task: asyncio.Task[None]) -> None:
        if self._release_task is task:
            self._release_task = None

    def _set_progress(self, task_id: str, value: float) -> None:
        record = self.records.get(task_id)
        if record is not None and record.task.status == "running":
            record.task.progress = min(0.99, max(0.0, value))
            if isinstance(value, SegmentProgress):
                record.task.raw_status = {**(record.task.raw_status or {}),
                    "current_segment": value.segment, "segment_count": value.count}
