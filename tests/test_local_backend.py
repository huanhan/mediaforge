import asyncio
import threading
from pathlib import Path

import pytest

from minimaxh3_service.backend import BackendError, OutputArtifact
from minimaxh3_service.config import Settings
from minimaxh3_service.local_backend import LocalGenerationBackend
from minimaxh3_service.models import GenerationRequest
from minimaxh3_service.workflow import GenerationPlan


class NotReadyRunner:
    def readiness(self) -> dict:
        return {
            "status": "not_ready",
            "backend": "native",
            "missing_models": ["fl2va"],
        }

    def generate(
        self,
        plan: GenerationPlan,
        output: OutputArtifact,
        cancel: threading.Event,
        progress,
    ) -> None:
        raise AssertionError("an unready backend must not start generation")

    def close(self) -> None:
        pass


def test_submit_rejects_when_local_models_are_not_ready(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = LocalGenerationBackend(
            Settings(data_root=tmp_path / "data", model_root=tmp_path / "models"),
            runner=NotReadyRunner(),
        )
        with pytest.raises(BackendError) as caught:
            await backend.submit(GenerationRequest(prompt="A local CUDA generation"))

        assert caught.value.status_code == 503
        assert caught.value.code == "not_ready"
        assert caught.value.detail["missing_models"] == ["fl2va"]

    asyncio.run(scenario())
