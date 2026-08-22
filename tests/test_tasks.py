import asyncio
import threading
from pathlib import Path

from minimaxh3_service.backend import OutputArtifact
from minimaxh3_service.models import GenerationRequest, UpscaleRequest
from minimaxh3_service.tasks import LocalTaskManager
from minimaxh3_service.workflow import GenerationPlan


class StubRunner:
    def __init__(self) -> None:
        self.closed = False
        self.released = 0

    def readiness(self) -> dict:
        return {"status": "ready"}

    def generate(
        self,
        plan: GenerationPlan,
        output: OutputArtifact,
        cancel: threading.Event,
        progress,
    ) -> None:
        assert plan.frame_count == 124
        assert not cancel.is_set()
        progress(0.5)
        output.path.parent.mkdir(parents=True, exist_ok=True)
        output.path.write_bytes(b"mp4")

    def close(self) -> None:
        self.closed = True

    def release_models(self) -> None:
        self.released += 1


class StubUpscaleRunner:
    def __init__(self, events: list[str] | None = None) -> None:
        self.closed = False
        self.released = 0
        self.events = events if events is not None else []

    def readiness(self) -> dict:
        return {"status": "ready"}

    def upscale(
        self,
        request: UpscaleRequest,
        output: OutputArtifact,
        cancel: threading.Event,
        progress,
    ) -> None:
        assert request.width == 1024
        assert request.height == 768
        assert not cancel.is_set()
        self.events.append("upscale-start")
        progress(0.5)
        output.path.parent.mkdir(parents=True, exist_ok=True)
        output.path.write_bytes(b"png")
        self.events.append("upscale-end")

    def close(self) -> None:
        self.closed = True

    def release_models(self) -> None:
        self.released += 1


def test_task_manager_runs_local_generation(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = StubRunner()

        def output_factory(task_id: str, _: str | None) -> OutputArtifact:
            return OutputArtifact("result.mp4", "", tmp_path / f"{task_id}.mp4")

        manager = LocalTaskManager(
            runner,
            output_factory,
            max_queue_size=2,
            allow_interrupt_running=False,
            idle_release_seconds=0.05,
        )
        await manager.start()
        submitted = await manager.submit(GenerationRequest(prompt="A local CUDA generation", seed=1))
        await manager.queue.join()
        task = manager.get(submitted.id)
        assert task is not None
        assert task.status == "succeeded"
        assert task.progress == 1.0
        assert task.outputs[0].path.read_bytes() == b"mp4"
        await asyncio.sleep(0.08)
        assert runner.released == 1
        await manager.close()
        assert runner.closed

    asyncio.run(scenario())


def test_task_arriving_during_idle_window_prevents_model_release(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = StubRunner()

        def output_factory(task_id: str, _: str | None) -> OutputArtifact:
            return OutputArtifact("result.mp4", "", tmp_path / f"{task_id}.mp4")

        manager = LocalTaskManager(
            runner,
            output_factory,
            max_queue_size=2,
            allow_interrupt_running=False,
            idle_release_seconds=0.1,
        )
        await manager.start()
        first = await manager.submit(GenerationRequest(prompt="first", seed=1))
        await manager.queue.join()
        await asyncio.sleep(0.03)
        second = await manager.submit(GenerationRequest(prompt="second", seed=2))
        await manager.queue.join()
        assert manager.get(first.id).status == "succeeded"
        assert manager.get(second.id).status == "succeeded"
        assert runner.released == 0
        await manager.close()

    asyncio.run(scenario())


def test_generation_and_upscale_share_queue_and_release_before_switch(tmp_path: Path) -> None:
    async def scenario() -> None:
        events: list[str] = []
        video_runner = StubRunner()
        upscale_runner = StubUpscaleRunner(events)

        def generation_output_factory(task_id: str, _: str | None) -> OutputArtifact:
            return OutputArtifact("result.mp4", "", tmp_path / f"{task_id}.mp4")

        def upscale_output_factory(task_id: str, _: str | None) -> OutputArtifact:
            return OutputArtifact("result.png", "", tmp_path / f"{task_id}.png")

        original_generate = video_runner.generate

        def record_generation(*args, **kwargs) -> None:
            events.append("generation-start")
            original_generate(*args, **kwargs)
            events.append("generation-end")

        video_runner.generate = record_generation
        original_release = video_runner.release_models

        def record_generation_release() -> None:
            events.append("generation-release")
            original_release()

        video_runner.release_models = record_generation_release

        manager = LocalTaskManager(
            video_runner,
            generation_output_factory,
            max_queue_size=4,
            allow_interrupt_running=False,
            idle_release_seconds=1.0,
            upscale_runner=upscale_runner,
            upscale_output_factory=upscale_output_factory,
            upscale_idle_release_seconds=1.0,
        )
        await manager.start()
        generation = await manager.submit(GenerationRequest(prompt="video", seed=1))
        upscale = await manager.submit_upscale(
            UpscaleRequest(file_id="input.png", width=1024, height=768, seed=2)
        )
        await manager.queue.join()

        assert manager.get(generation.id).status == "succeeded"
        assert manager.get(upscale.id).status == "succeeded"
        assert events == [
            "generation-start",
            "generation-end",
            "generation-release",
            "upscale-start",
            "upscale-end",
        ]
        assert video_runner.released == 1
        assert upscale_runner.released == 0
        await manager.close()

    asyncio.run(scenario())


def test_upscale_idle_timeout_releases_only_upscale_runner(tmp_path: Path) -> None:
    async def scenario() -> None:
        video_runner = StubRunner()
        upscale_runner = StubUpscaleRunner()

        def output_factory(task_id: str, _: str | None) -> OutputArtifact:
            return OutputArtifact("result.mp4", "", tmp_path / f"{task_id}.mp4")

        def upscale_output_factory(task_id: str, _: str | None) -> OutputArtifact:
            return OutputArtifact("result.png", "", tmp_path / f"{task_id}.png")

        manager = LocalTaskManager(
            video_runner,
            output_factory,
            max_queue_size=2,
            allow_interrupt_running=False,
            idle_release_seconds=1.0,
            upscale_runner=upscale_runner,
            upscale_output_factory=upscale_output_factory,
            upscale_idle_release_seconds=0.05,
        )
        await manager.start()
        submitted = await manager.submit_upscale(
            UpscaleRequest(file_id="input.png", width=1024, height=768)
        )
        await manager.queue.join()

        for _ in range(100):
            if upscale_runner.released == 1:
                break
            await asyncio.sleep(0.01)

        assert manager.get(submitted.id).status == "succeeded"
        assert upscale_runner.released == 1
        assert video_runner.released == 0
        await manager.close()

    asyncio.run(scenario())
