import asyncio
import threading
from pathlib import Path

from minimaxh3_service.backend import OutputArtifact
from minimaxh3_service.models import GenerationRequest
from minimaxh3_service.tasks import LocalTaskManager
from minimaxh3_service.workflow import GenerationPlan


class StubRunner:
    def __init__(self) -> None:
        self.closed = False

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
        )
        await manager.start()
        submitted = await manager.submit(GenerationRequest(prompt="A local CUDA generation", seed=1))
        await manager.queue.join()
        task = manager.get(submitted.id)
        assert task is not None
        assert task.status == "succeeded"
        assert task.progress == 1.0
        assert task.outputs[0].path.read_bytes() == b"mp4"
        await manager.close()
        assert runner.closed

    asyncio.run(scenario())
