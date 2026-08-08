from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import UploadFile
from starlette.testclient import TestClient

from minimaxh3_service.api import create_app
from minimaxh3_service.backend import (
    BackendSubmission,
    BackendTask,
    OutputArtifact,
    StoredAsset,
)
from minimaxh3_service.config import Settings
from minimaxh3_service.models import GenerationRequest


class FakeLocalBackend:
    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        self.tasks: dict[str, BackendTask] = {}
        self.closed = False

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def ready(self) -> dict[str, Any]:
        return {"status": "ready", "backend": "native", "devices": [{"type": "cuda"}]}

    async def upload(
        self,
        upload: UploadFile,
        *,
        kind: Literal["image", "video", "audio"],
    ) -> StoredAsset:
        return StoredAsset(
            file_id="minimaxh3/20260808/start-123.png",
            filename="start-123.png",
            subfolder="minimaxh3/20260808",
            kind=kind,
        )

    async def submit(self, request: GenerationRequest) -> BackendSubmission:
        self.tasks["task-123"] = BackendTask(id="task-123", status="queued", progress=0.0)
        return BackendSubmission(
            id="task-123",
            queue_number=1,
            frame_count=124,
            effective_duration_seconds=124 / 24,
        )

    async def get_task(self, task_id: str) -> BackendTask | None:
        return self.tasks.get(task_id)

    async def cancel(self, task_id: str) -> BackendTask | None:
        task = self.tasks.get(task_id)
        if task is not None:
            task.status = "canceled"
        return task

    def resolve_file(
        self,
        *,
        filename: str,
        subfolder: str,
        file_type: Literal["input", "output", "temp"],
    ) -> Path:
        return self.output_file


def make_client(tmp_path: Path, *, api_key: str | None = None) -> tuple[TestClient, FakeLocalBackend]:
    output_file = tmp_path / "MiniMax_H3_00001_.mp4"
    output_file.write_bytes(b"video")
    fake = FakeLocalBackend(output_file)
    app = create_app(
        Settings(api_key=api_key, data_root=tmp_path / "data"),
        backend_factory=lambda _: fake,
    )
    return TestClient(app), fake


def test_upload_submit_query_and_download(tmp_path: Path) -> None:
    client, fake = make_client(tmp_path)
    with client:
        uploaded = client.post(
            "/v1/files",
            data={"kind": "image"},
            files={"file": ("start.png", b"png", "image/png")},
        )
        assert uploaded.status_code == 201
        file_id = uploaded.json()["file_id"]

        submitted = client.post(
            "/v1/generations",
            json={
                "prompt": "A slow camera move",
                "mode": "fl2v",
                "first_frame": file_id,
                "seed": 7,
            },
        )
        assert submitted.status_code == 202
        assert submitted.json()["id"] == "task-123"
        assert submitted.json()["frame_count"] == 124
        assert client.get("/v1/tasks/task-123").json()["status"] == "queued"

        fake.tasks["task-123"] = BackendTask(
            id="task-123",
            status="succeeded",
            progress=1.0,
            outputs=[
                OutputArtifact(
                    filename="MiniMax_H3_00001_.mp4",
                    subfolder="video",
                    path=fake.output_file,
                )
            ],
        )
        completed = client.get("/v1/tasks/task-123")
        assert completed.status_code == 200
        assert completed.json()["status"] == "succeeded"
        download_url = completed.json()["outputs"][0]["download_url"]
        assert client.get(download_url).content == b"video"


def test_ready_reports_native_backend(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["backend"] == "native"
        assert client.get("/v1/models").json()["backend"] == "native"


def test_cancel_queued_task(tmp_path: Path) -> None:
    client, fake = make_client(tmp_path)
    fake.tasks["task-123"] = BackendTask(id="task-123", status="queued")
    with client:
        response = client.delete("/v1/tasks/task-123")
        assert response.status_code == 200
        assert response.json()["status"] == "canceled"


def test_api_key_is_optional_but_enforced_when_configured(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, api_key="secret")
    with client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"X-API-Key": "secret"}).status_code == 200
        assert client.get(
            "/v1/models", headers={"Authorization": "Bearer secret"}
        ).status_code == 200


def test_rejects_wrong_upload_extension(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        response = client.post(
            "/v1/files",
            data={"kind": "image"},
            files={"file": ("clip.mp4", b"video", "video/mp4")},
        )
        assert response.status_code == 415
