import asyncio
import threading
from pathlib import Path

import pytest

from mediaforge_service.backend import BackendError, OutputArtifact
from mediaforge_service.config import Settings
from mediaforge_service.local_backend import LocalGenerationBackend
from mediaforge_service.models import GenerationRequest
from mediaforge_service.workflow import GenerationPlan


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


class ReadyRunner(NotReadyRunner):
    def readiness(self) -> dict:
        return {"status": "ready", "backend": "native"}


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


def test_submit_ref2v_rejects_missing_acceleration_lora(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = Settings(
            data_root=tmp_path / "data",
            model_root=tmp_path / "models",
            enable_acceleration_lora=True,
        )
        ref2va = settings.model_paths["ref2va"]
        ref2va.parent.mkdir(parents=True)
        ref2va.touch()
        backend = LocalGenerationBackend(settings, runner=ReadyRunner())

        with pytest.raises(BackendError) as caught:
            await backend.submit(
                GenerationRequest(
                    prompt="A reference-driven scene",
                    mode="ref2v",
                    reference_images=["reference.png"],
                )
            )

        assert caught.value.status_code == 503
        assert caught.value.code == "model_missing"
        assert "acceleration LoRA" in str(caught.value)
        assert "--with-ref2v" in str(caught.value)
        assert "--with-acceleration-lora" not in str(caught.value)

    asyncio.run(scenario())
