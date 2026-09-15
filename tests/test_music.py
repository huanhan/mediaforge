import asyncio
import io
import json
import threading
import wave

import httpx
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from minimaxh3_service.api import create_app
from minimaxh3_service.backend import BackendError
from minimaxh3_service.config import Settings
from minimaxh3_service.local_backend import LocalGenerationBackend
from minimaxh3_service.models import MusicRequest
from minimaxh3_service.music import MusicRunner
from minimaxh3_service.storage import LocalAssetStore
from minimaxh3_service.tasks import LocalTaskManager
from minimaxh3_service.ui import VideoClient


def wav_bytes():
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setparams((2, 2, 32000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0" * 1280)
    return data.getvalue()


class IdleRunner:
    def readiness(self):
        return {"status": "not_ready"}

    def close(self):
        pass

    def release_models(self):
        pass


def test_protocol_and_wav(tmp_path):
    captured = []
    def handler(request):
        captured.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "MiniMaxAI/MiniMax-Music3"}]})
        return httpx.Response(200, content=wav_bytes())
    runner = MusicRunner(Settings(music_base_url="http://music:8001", music_api_key="test-key"),
                         transport=httpx.MockTransport(handler))
    assert runner.readiness()["status"] == "ready"
    artifact = LocalAssetStore(tmp_path / "in", tmp_path / "out", 1000).music_output_path("abc")
    runner.generate(MusicRequest(prompt="folk", lyrics="[Verse]\nHello"), artifact, threading.Event(), lambda _: None)
    payload = json.loads(captured[-1].content)
    assert captured[-1].url.path == "/v1/audio/speech"
    assert captured[-1].headers["Authorization"] == "Bearer test-key"
    assert payload == dict(model="MiniMaxAI/MiniMax-Music3", input="[Verse]\nHello",
                           instructions="folk", seed=7, max_new_tokens=750, response_format="wav", stream=False)
    assert artifact.path.read_bytes() == wav_bytes()


@pytest.mark.parametrize("content,status", [(b"not audio", 200), (wav_bytes()[:-10], 200), (b"error", 500), (b"", 302)])
def test_bad_upstream_cleans_output(tmp_path, content, status):
    runner = MusicRunner(Settings(music_base_url="http://music"),
                         transport=httpx.MockTransport(lambda _: httpx.Response(status, content=content)))
    output = LocalAssetStore(tmp_path / "in", tmp_path / "out", 1000).music_output_path("bad")
    with pytest.raises((ValueError, wave.Error, EOFError, BackendError)):
        runner.generate(MusicRequest(prompt="folk", lyrics="hello"), output, threading.Event(), lambda _: None)
    assert not output.path.exists()


@pytest.mark.parametrize("update", [{"prompt": " "}, {"duration_seconds": 301},
                                     {"duration_seconds": 9}, {"seed": -1}, {"is_instrumental": True}])
def test_music_validation(update):
    with pytest.raises(ValidationError):
        MusicRequest.model_validate({"prompt": "folk", "lyrics": "hello", **update})


def test_readiness_missing_and_wrong_model():
    assert MusicRunner(Settings()).readiness()["status"] == "not_ready"
    runner = MusicRunner(Settings(music_base_url="http://music"), transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"data": [{"id": "other"}]})))
    assert runner.readiness()["status"] == "not_ready"


def test_api_song_download_and_auth(tmp_path):
    settings = Settings(data_root=tmp_path, api_key="secret", music_base_url="http://music")
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": settings.music_model}]})
        return httpx.Response(200, content=wav_bytes())
    backend = LocalGenerationBackend(settings, runner=IdleRunner(), upscale_runner=IdleRunner(),
                                     removal_runner=IdleRunner(), music_runner=MusicRunner(settings, transport=httpx.MockTransport(handler)))
    with TestClient(create_app(settings=settings, backend_factory=lambda _: backend)) as client:
        payload = dict(prompt="folk", lyrics="hello")
        assert client.post("/v1/music/generations", json=payload).status_code == 401
        client.headers["X-API-Key"] = "secret"
        submitted = client.post("/v1/music/generations", json=payload)
        assert submitted.status_code == 202
        client.portal.call(backend.tasks.queue.join)
        task = client.get("/v1/tasks/" + submitted.json()["id"]).json()
        assert task["status"] == "succeeded"
        assert task["kind"] == "music"
        output = task["outputs"][0]
        response = client.get(output["download_url"])
        assert response.status_code == 200
        assert response.content == wav_bytes()
        assert "audio" in response.headers["content-type"]
        assert client.post("/v1/music/generations", json={**payload, "lyrics": ""}).status_code == 422


def test_disabled_music_returns_503(tmp_path):
    settings = Settings(data_root=tmp_path, music_model_path=tmp_path / "missing.safetensors")
    backend = LocalGenerationBackend(settings, runner=IdleRunner(), upscale_runner=IdleRunner(), removal_runner=IdleRunner())
    with TestClient(create_app(settings=settings, backend_factory=lambda _: backend)) as client:
        assert client.post("/v1/music/generations", json=dict(prompt="folk", lyrics="hello")).status_code == 503


def test_queue_cancel_and_failure(tmp_path):
    async def scenario():
        class BrokenMusic(IdleRunner):
            def generate(self, request, output, cancel, progress):
                output.path.write_bytes(b"partial")
                raise ValueError("upstream failed")
        store = LocalAssetStore(tmp_path / "in", tmp_path / "out", 1000)
        manager = LocalTaskManager(IdleRunner(), store.output_path, max_queue_size=2,
                                   allow_interrupt_running=True, music_runner=BrokenMusic(),
                                   music_output_factory=store.music_output_path)
        request = MusicRequest(prompt="folk", lyrics="hello")
        first = await manager.submit_music(request)
        assert manager.cancel(first.id).status == "canceled"
        second = await manager.submit_music(request)
        with pytest.raises(BackendError) as full:
            await manager.submit_music(request)
        assert full.value.status_code == 429
        await manager.start()
        await manager.queue.join()
        assert manager.get(first.id).status == "canceled"
        assert manager.get(second.id).status == "failed"
        assert not manager.records[second.id].output.path.exists()
        manager.records[second.id].task.status = "running"
        with pytest.raises(BackendError) as running:
            manager.cancel(second.id)
        assert running.value.status_code == 409
        manager.records[second.id].task.status = "failed"
        await manager.close()
    asyncio.run(scenario())


def test_ui_client_submits_music():
    def handler(request):
        assert request.url.path == "/v1/music/generations"
        assert json.loads(request.content)["lyrics"] == "hello"
        return httpx.Response(202, json={"id": "song", "status": "queued", "queue_number": 1})
    client = VideoClient("http://localhost:8000", transport=httpx.MockTransport(handler))
    assert client.submit_music(dict(prompt="folk", lyrics="hello"))["id"] == "song"


def test_native_request_limits_and_optional_lyrics(tmp_path):
    from minimaxh3_service.native_music import NativeMusicRunner
    runner = NativeMusicRunner(Settings(music_model_path=tmp_path / "missing"), None)
    assert runner.readiness()["status"] == "not_ready"
    assert MusicRequest(prompt="cheerful").lyrics == ""
    runner.validate_request(MusicRequest(prompt="cheerful", duration_seconds=120))
    with pytest.raises(BackendError) as error:
        runner.validate_request(MusicRequest(prompt="cheerful", duration_seconds=121))
    assert error.value.status_code == 422
    with pytest.raises(BackendError):
        MusicRunner(Settings()).validate_request(MusicRequest(prompt="cheerful"))


def test_native_pcm_and_cancellation(tmp_path):
    from types import SimpleNamespace
    from minimaxh3_service.native_music import NativeMusicRunner
    from minimaxh3_service.tasks import GenerationCanceled
    torch = pytest.importorskip("torch")
    seen = {}
    class Clip:
        def tokenize(self, prompt, lyrics):
            seen["lyrics"] = lyrics
            return prompt
        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.ones(1, 2, 3), {}]]
    class Vae:
        def throw_exception_if_invalid(self):
            pass
        def decode(self, latent):
            return torch.ones(1, 100, 2) * 0.25
    def sample(model, noise, steps, cfg, sampler, scheduler, positive, negative, latent, **kwargs):
        seen["latent"] = latent.shape
        kwargs["callback"](0, None, None, steps)
        return latent
    core = SimpleNamespace(torch=torch,
        sd=SimpleNamespace(load_checkpoint_guess_config=lambda *a, **kw: (object(), Clip(), Vae())),
        sample=SimpleNamespace(prepare_noise=lambda latent, seed: torch.zeros_like(latent), sample=sample),
        model_management=SimpleNamespace(unload_all_models=lambda: None, soft_empty_cache=lambda: None))
    provider = SimpleNamespace(_import_core=lambda **kw: core)
    runner = NativeMusicRunner(Settings(), provider)
    output = LocalAssetStore(tmp_path / "in", tmp_path / "out", 1000).music_output_path("native")
    runner.generate(MusicRequest(prompt="cheerful", duration_seconds=10), output, threading.Event(), lambda _: None)
    assert seen["lyrics"] == "[instrumental]"
    assert seen["latent"] == (1, 8, 16, int(10 * 44100 / 4096))
    with wave.open(str(output.path), "rb") as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (2, 44100, 100)
    cancel = threading.Event()
    def progress(value):
        if value >= 0.12:
            cancel.set()
    with pytest.raises(GenerationCanceled):
        runner.generate(MusicRequest(prompt="cheerful", duration_seconds=10), output, cancel, progress)
    assert not output.path.exists()
    assert runner._model is None
