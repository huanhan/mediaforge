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
    UpscaleSubmission,
)
from minimaxh3_service.config import Settings
from minimaxh3_service.models import GenerationRequest, UpscaleRequest, VideoEditRequest


class FakeLocalBackend:
    async def submit_edit(self, request: VideoEditRequest) -> BackendSubmission:
        self.edit_request = request
        return await self.submit(request)

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        self.upscale_output_file = output_file.with_name("restored-image.png")
        self.upscale_video_output_file = output_file.with_name("restored-video.mp4")
        self.upscale_output_file.write_bytes(b"png")
        self.upscale_video_output_file.write_bytes(b"upscaled-video")
        self.tasks: dict[str, BackendTask] = {}
        self.closed = False

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    async def ready(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "backend": "native",
            "devices": [{"type": "cuda"}],
            "upscale": {"status": "ready", "backend": "seedvr2"},
        }

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
        self.tasks["task-123"] = BackendTask(
            id="task-123", status="queued", kind="generation", progress=0.0
        )
        return BackendSubmission(
            id="task-123",
            queue_number=1,
            frame_count=124,
            effective_duration_seconds=124 / 24,
        )

    async def submit_upscale(self, request: UpscaleRequest) -> UpscaleSubmission:
        self.tasks["upscale-123"] = BackendTask(
            id="upscale-123", status="queued", kind="upscale", progress=0.0
        )
        return UpscaleSubmission(
            id="upscale-123",
            queue_number=1,
            width=request.width,
            height=request.height,
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
        if filename == self.upscale_output_file.name:
            return self.upscale_output_file
        if filename == self.upscale_video_output_file.name:
            return self.upscale_video_output_file
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
            kind="generation",
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


def test_upscale_submit_query_and_png_download(tmp_path: Path) -> None:
    client, fake = make_client(tmp_path)
    with client:
        submitted = client.post(
            "/v1/upscales",
            json={
                "file_id": "minimaxh3/20260808/source.png",
                "width": 1024,
                "height": 768,
                "seed": 7,
                "filename_prefix": "restored/image",
            },
        )
        assert submitted.status_code == 202
        assert submitted.json() == {
            "id": "upscale-123",
            "status": "queued",
            "queue_number": 1,
            "width": 1024,
            "height": 768,
        }

        queued = client.get("/v1/tasks/upscale-123")
        assert queued.status_code == 200
        assert queued.json()["kind"] == "upscale"
        assert queued.json()["status"] == "queued"

        fake.tasks["upscale-123"] = BackendTask(
            id="upscale-123",
            status="succeeded",
            kind="upscale",
            progress=1.0,
            outputs=[
                OutputArtifact(
                    filename="restored-image.png",
                    subfolder="upscale",
                    path=fake.upscale_output_file,
                )
            ],
        )
        completed = client.get("/v1/tasks/upscale-123")
        assert completed.status_code == 200
        payload = completed.json()
        assert payload["kind"] == "upscale"
        assert payload["status"] == "succeeded"
        assert payload["outputs"][0]["filename"] == "restored-image.png"
        assert payload["outputs"][0]["type"] == "output"
        assert client.get(payload["outputs"][0]["download_url"]).content == b"png"


def test_upscale_video_submit_query_and_mp4_download(tmp_path: Path) -> None:
    client, fake = make_client(tmp_path)
    with client:
        submitted = client.post(
            "/v1/upscales",
            json={
                "file_id": "minimaxh3/20260822/generated.mp4",
                "width": 1024,
                "height": 768,
                "seed": 7,
                "filename_prefix": "restored/video",
            },
        )
        assert submitted.status_code == 202
        assert submitted.json()["id"] == "upscale-123"

        fake.tasks["upscale-123"] = BackendTask(
            id="upscale-123",
            status="succeeded",
            kind="upscale",
            progress=1.0,
            outputs=[
                OutputArtifact(
                    filename="restored-video.mp4",
                    subfolder="upscale",
                    path=fake.upscale_video_output_file,
                )
            ],
        )
        completed = client.get("/v1/tasks/upscale-123")
        assert completed.status_code == 200
        payload = completed.json()
        assert payload["kind"] == "upscale"
        assert payload["status"] == "succeeded"
        assert payload["outputs"][0]["filename"] == "restored-video.mp4"
        assert payload["outputs"][0]["type"] == "output"
        assert client.get(payload["outputs"][0]["download_url"]).content == b"upscaled-video"


def test_ready_reports_native_backend(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["backend"] == "native"
        assert client.get("/v1/models").json()["backend"] == "native"


def test_cancel_queued_task(tmp_path: Path) -> None:
    client, fake = make_client(tmp_path)
    fake.tasks["task-123"] = BackendTask(
        id="task-123", status="queued", kind="generation"
    )
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


def _operation(spec: dict[str, Any], path: str, method: str) -> dict[str, Any]:
    return spec["paths"][path][method]


def _request_examples(
    spec: dict[str, Any], path: str, method: str, schema_name: str
) -> list[dict[str, Any]]:
    schema = spec["components"]["schemas"][schema_name]
    examples = schema.get("examples")
    if not examples:
        request_body = _operation(spec, path, method).get("requestBody", {})
        content = request_body.get("content", {}).get("application/json", {})
        examples = content.get("examples")
    if isinstance(examples, dict):
        examples = list(examples.values())
    values: list[dict[str, Any]] = []
    for example in examples or []:
        value = (
            example["value"]
            if isinstance(example, dict) and "value" in example
            else example
        )
        if isinstance(value, dict):
            values.append(value)
    return values


def test_openapi_documents_routes_auth_and_media_contract(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200

    spec = openapi_response.json()
    assert spec["info"]["title"]
    description = f"{spec['info']['title']} {spec['info'].get('description', '')}".lower()
    assert "video" in description
    assert "generation" in description
    assert "image" in description
    assert "upscal" in description

    expected_operation_ids = {
        ("/healthz", "get"): "health_check",
        ("/readyz", "get"): "readiness_check",
        ("/v1/models", "get"): "list_models",
        ("/v1/files", "post"): "upload_file",
        ("/v1/generations", "post"): "create_generation",
        ("/v1/upscales", "post"): "create_upscale",
        ("/v1/tasks/{task_id}", "get"): "get_task",
        ("/v1/tasks/{task_id}", "delete"): "cancel_task",
        ("/v1/tasks/{task_id}/events", "get"): "stream_task_events",
        ("/v1/files/content", "get"): "download_file",
    }
    actual_operation_ids = {
        key: _operation(spec, *key)["operationId"] for key in expected_operation_ids
    }
    assert actual_operation_ids == expected_operation_ids
    assert len(set(actual_operation_ids.values())) == len(actual_operation_ids)

    schemes = spec["components"]["securitySchemes"]
    assert {"ApiKeyAuth", "BearerAuth"}.issubset(schemes)
    for (path, method), operation_id in expected_operation_ids.items():
        operation = _operation(spec, path, method)
        if path.startswith("/v1/"):
            security = operation.get("security", [])
            security_names = {name for requirement in security for name in requirement}
            assert {"ApiKeyAuth", "BearerAuth"}.issubset(security_names), operation_id
        else:
            assert not operation.get("security"), operation_id

    generation_examples = _request_examples(
        spec, "/v1/generations", "post", "GenerationRequest"
    )
    assert any(example.get("prompt") and example.get("mode") for example in generation_examples)
    upscale_examples = _request_examples(spec, "/v1/upscales", "post", "UpscaleRequest")
    assert any(
        example.get("file_id") and example.get("width") and example.get("height")
        for example in upscale_examples
    )

    event_response = _operation(spec, "/v1/tasks/{task_id}/events", "get")["responses"]["200"]
    event_content = event_response["content"]["text/event-stream"]
    assert event_content["schema"] == {"type": "string"}

    download_response = _operation(spec, "/v1/files/content", "get")["responses"]["200"]
    binary_content = download_response["content"]["application/octet-stream"]
    assert binary_content["schema"] == {"type": "string", "format": "binary"}


def test_openapi_documents_business_error_statuses(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    with client:
        response = client.get("/openapi.json")
        assert response.status_code == 200
    spec = response.json()

    expected_errors = {
        ("/readyz", "get"): {"503"},
        ("/v1/files", "post"): {"401", "413", "415"},
        ("/v1/generations", "post"): {"401", "429", "503"},
        ("/v1/upscales", "post"): {"401", "415", "429", "503"},
        ("/v1/tasks/{task_id}", "get"): {"401", "404"},
        ("/v1/tasks/{task_id}", "delete"): {"401", "404"},
        ("/v1/tasks/{task_id}/events", "get"): {"401", "404"},
        ("/v1/files/content", "get"): {"401", "404"},
    }
    for route, statuses in expected_errors.items():
        responses = _operation(spec, *route)["responses"]
        assert statuses.issubset(responses), route
        for status_code in statuses:
            assert responses[status_code].get("description"), (route, status_code)


def test_independent_edit_contract(tmp_path):
    client, fake = make_client(tmp_path)
    with client:
        payload = {"prompt": "Replace background", "source_video": {"file_id": "source.mp4"}}
        assert client.post("/v1/edits", json=payload).status_code == 202
        assert isinstance(fake.edit_request, VideoEditRequest)
        assert fake.edit_request.audio_mode == "preserve_source"
        for extra in ({"mode": "edit"}, {"duration_seconds": 8}, {"first_frame": "f.png"},
                      {"reference_videos": []}, {"reference_audios": []}):
            assert client.post("/v1/edits", json={**payload, **extra}).status_code == 422
        assert client.post("/v1/edits", json={"prompt": "x"}).status_code == 422
        for extra in ({"mode": "edit"}, {"source_video": {"file_id": "s.mp4"}},
                      {"audio_mode": "generate"}):
            assert client.post("/v1/generations", json={"prompt": "x", **extra}).status_code == 422
        schema = client.get("/openapi.json").json()
        for path, model in (("/v1/edits", VideoEditRequest), ("/v1/generations", GenerationRequest)):
            content = schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]
            assert content["schema"]["$ref"].endswith("/" + model.__name__)
            for example in content["examples"].values():
                model.model_validate(example["value"])
        assert "edit" not in schema["components"]["schemas"]["GenerationMode"]["enum"]


def test_edit_size_presets_http_contract(tmp_path):
    client, fake = make_client(tmp_path)
    with client:
        body = {"prompt": "x", "source_video": {"file_id": "s.mp4"},
                "resolution": "480p", "aspect_ratio": "9:16"}
        assert client.post("/v1/edits", json=body).status_code == 202
        assert (fake.edit_request.width, fake.edit_request.height) == (480, 832)
        assert client.post("/v1/edits", json={**body, "width": 608, "height": 352}).status_code == 202
        assert (fake.edit_request.width, fake.edit_request.height) == (608, 352)
        for extra in ({"width": 608}, {"resolution": "invalid"}, {"aspect_ratio": "invalid"}):
            assert client.post("/v1/edits", json={**body, **extra}).status_code == 422
