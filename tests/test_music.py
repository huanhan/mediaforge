import asyncio
import json
import subprocess
import sys
import threading
import wave
from types import SimpleNamespace

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
from minimaxh3_service.tasks import GenerationCanceled, LocalTaskManager
from minimaxh3_service.ui import VideoClient


class IdleRunner:
    def readiness(self):
        return {"status": "not_ready"}

    def close(self):
        pass

    def release_models(self):
        pass


@pytest.fixture
def music(tmp_path):
    # A real child process exercises the transport/lifetime; no model downloads in tests.
    script = tmp_path / "fake_worker.py"
    script.write_text("""
import argparse, json, os, time, wave
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--request', type=Path)
p.add_argument('--output', type=Path)
p.add_argument('--progress', type=Path)
args, extra = p.parse_known_args()
Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))
request = json.loads(args.request.read_text(encoding='utf-8'))
assert os.environ['HF_HUB_OFFLINE'] == '1'
assert '--offload-ar' in extra
args.progress.write_text('{"progress": 0.72}', encoding='utf-8')
if request['prompt'] == 'slow':
    time.sleep(30)
if request['prompt'] == 'fail':
    args.output.write_bytes(b'partial')
    raise RuntimeError('deliberate worker failure')
with wave.open(str(args.output), 'wb') as wav:
    wav.setparams((2, 2, 44100 if request['prompt'] == 'bad-rate' else 48000, 0, 'NONE', 'not compressed'))
    wav.writeframes(b'\\0' * 1920)
if request['prompt'] == 'truncated':
    args.output.write_bytes(args.output.read_bytes()[:-4])
metadata = dict(schema_version=1, model='m-a-p/YuE2-3B', sample_rate=48000,
                duration_seconds=0.01, truncated=dict(abc=False, semantic=True), request=request)
if request['prompt'] == 'bad-metadata':
    metadata['sample_rate'] = 32000
args.output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
if request['cot'] != 'off':
    args.output.with_suffix('.abc').write_text('X:1\\nK:C\\nCDEF|', encoding='utf-8')
""", encoding="utf-8")

    class ScriptRunner(MusicRunner):
        def _command(self):
            return [sys.executable, str(script)]

        def readiness(self):
            return {"status": "ready", "backend": "yue2", "model": "m-a-p/YuE2-3B"}

    settings = Settings(data_root=tmp_path, api_key="secret", music_timeout_seconds=10)
    output = LocalAssetStore(tmp_path / "in", tmp_path / "out", 1000).music_output_path("song")
    return ScriptRunner(settings), output


def test_worker_artifacts_and_utf8(music):
    runner, output = music
    request = MusicRequest(prompt="温暖钢琴", lyrics="[Verse]\n清晨的歌", abc="X:1\nK:C\nCDEF|", seed=42)
    artifacts = runner.generate(request, output, threading.Event(), lambda _: None)
    assert [a.path.suffix for a in artifacts] == [".wav", ".json", ".abc"]
    assert MusicRunner._validate_audio(output.path, 30) == .01
    metadata = json.loads(artifacts[1].path.read_text(encoding="utf-8"))
    assert metadata["request"] == request.model_dump()
    assert metadata["truncated"]["semantic"] is True
    assert not list(output.path.parent.glob(".yue2-*"))


@pytest.mark.parametrize("prompt", ["fail", "bad-rate", "truncated", "bad-metadata"])
def test_failed_worker_cleans_outputs(music, prompt):
    runner, output = music
    with pytest.raises((RuntimeError, ValueError)):
        runner.generate(MusicRequest(prompt=prompt), output, threading.Event(), lambda _: None)
    assert not output.path.exists()
    assert not list(output.path.parent.glob(".yue2-*"))
    assert not output.path.with_suffix(".json").exists()


@pytest.mark.parametrize("mode", ["cancel", "timeout"])
def test_running_worker_is_reaped(music, monkeypatch, mode):
    psutil = pytest.importorskip("psutil")
    runner, output = music
    pid_file = runner.settings.data_root / "fake_worker.pid"
    processes = []
    original = subprocess.Popen

    def capture(*a, **kw):
        process = original(*a, **kw)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)
    cancel = threading.Event()
    if mode == "timeout":
        runner.settings = Settings(music_timeout_seconds=.2)
    timer = threading.Timer(.4, cancel.set) if mode == "cancel" else None
    if timer:
        timer.start()
    try:
        with pytest.raises(GenerationCanceled if mode == "cancel" else TimeoutError):
            runner.generate(MusicRequest(prompt="slow"), output, cancel, lambda _: None)
    finally:
        if timer:
            timer.cancel()
            timer.join()
    assert processes and processes[0].poll() is not None
    if pid_file.exists():
        assert not psutil.pid_exists(int(pid_file.read_text()))
    assert not output.path.exists()
    assert not list(output.path.parent.glob(".yue2-*"))


@pytest.mark.parametrize("update", [
    {"prompt": " "}, {"duration_seconds": 301}, {"duration_seconds": 9}, {"seed": -1},
    {"is_instrumental": True}, {"cot": "bad"}, {"abc": " "}, {"abc": "X:1", "cot": "off"},
    {"cfg_scale": float("nan")}, {"cfg_scale": float("inf")}, {"cfg_scale": 21},
])
def test_music_validation(update):
    with pytest.raises(ValidationError):
        MusicRequest.model_validate({"prompt": "folk", "lyrics": "hello", **update})


def test_settings_and_readiness(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_MUSIC_PYTHON", str(tmp_path / "python"))
    monkeypatch.setenv("H3_MUSIC_MODEL_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("H3_MUSIC_DEVICE", "cuda:1")
    monkeypatch.setenv("H3_MUSIC_MEMORY_BUDGET_GIB", "24")
    monkeypatch.setenv("H3_MUSIC_OFFLOAD_AR", "false")
    settings = Settings.from_env()
    assert settings.music_device == "cuda:1"
    assert settings.music_memory_budget_gib == 24
    assert settings.music_offload_ar is False
    ready = MusicRunner(settings).readiness()
    assert ready["status"] == "not_ready"
    assert ready["backend"] == "yue2"
    assert ready["missing"]


@pytest.mark.parametrize("stdout,status", [('{"status":"ready"}\n\x1b[0m', "ready"), ("invalid output", "not_ready")])
def test_readiness_probe_handles_console_suffix_and_invalid_output(monkeypatch, stdout, status):
    from pathlib import Path
    from minimaxh3_service import music as module
    monkeypatch.setattr(module, "missing_model_files", lambda _: [])
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=stdout))
    runner = MusicRunner(Settings(music_python=Path(sys.executable)))
    assert runner.readiness()["status"] == status


@pytest.mark.parametrize("values", [
    {"music_memory_budget_gib": 2}, {"music_memory_budget_gib": float("nan")},
    {"music_device": "cuda:-1"}, {"music_timeout_seconds": float("inf")},
])
def test_invalid_settings(values):
    with pytest.raises(ValueError):
        Settings(**values)


def test_api_song_download_and_auth(music):
    runner, _ = music
    settings = runner.settings
    idle = IdleRunner()
    backend = LocalGenerationBackend(settings, runner=idle, upscale_runner=idle,
                                     removal_runner=idle, music_runner=runner, depth_runner=idle,
                                     skeleton_runner=idle, beat_runner=idle, transcription_runner=idle,
                                     alignment_runner=idle)
    with TestClient(create_app(settings=settings, backend_factory=lambda _: backend)) as client:
        payload = dict(prompt="folk", lyrics="[Verse]\nhello")
        assert client.post("/v1/music/generations", json=payload).status_code == 401
        client.headers["X-API-Key"] = "secret"
        catalog = client.get("/v1/models").json()["capabilities"]["music"]
        assert catalog["available"] and catalog["backend"] == "yue2"
        submitted = client.post("/v1/music/generations", json=payload)
        assert submitted.status_code == 202
        client.portal.call(backend.tasks.queue.join)
        task = client.get("/v1/tasks/" + submitted.json()["id"]).json()
        assert task["status"] == "succeeded"
        assert task["kind"] == "music"
        assert len(task["outputs"]) == 3
        for output in task["outputs"]:
            response = client.get(output["download_url"])
            assert response.status_code == 200
            if output["filename"].endswith(".wav"):
                assert "audio" in response.headers["content-type"]
            if output["filename"].endswith(".json"):
                assert response.json()["truncated"]["semantic"] is True
        assert client.post("/v1/music/generations", json={**payload, "abc": "X:1", "cot": "off"}).status_code == 422


def test_disabled_music_returns_503(tmp_path):
    settings = Settings(data_root=tmp_path, music_model_path=tmp_path / "missing")
    backend = LocalGenerationBackend(settings, runner=IdleRunner(), upscale_runner=IdleRunner(),
                                     removal_runner=IdleRunner())
    with TestClient(create_app(settings=settings, backend_factory=lambda _: backend)) as client:
        response = client.post("/v1/music/generations", json=dict(prompt="folk"))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "music_not_ready"


def test_queue_cancel_and_failure(tmp_path):
    async def scenario():
        class BrokenMusic(IdleRunner):
            def generate(self, request, output, cancel, progress):
                output.path.write_bytes(b"partial")
                raise ValueError("worker failed")
        store = LocalAssetStore(tmp_path / "in", tmp_path / "out", 1000)
        manager = LocalTaskManager(IdleRunner(), store.output_path, max_queue_size=2,
                                   allow_interrupt_running=True, music_runner=BrokenMusic(),
                                   music_output_factory=store.music_output_path)
        request = MusicRequest(prompt="folk")
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
        await manager.close()
    asyncio.run(scenario())


def test_ui_client_submits_music():
    def handler(request):
        assert request.url.path == "/v1/music/generations"
        assert json.loads(request.content)["lyrics"] == "hello"
        return httpx.Response(202, json={"id": "song", "status": "queued", "queue_number": 1})
    client = VideoClient("http://localhost:8000", transport=httpx.MockTransport(handler))
    assert client.submit_music(dict(prompt="folk", lyrics="hello"))["id"] == "song"


def test_worker_maps_real_pipeline_contract_and_limits_audio(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from minimaxh3_service.music_worker import generate
    captured = {}

    class Pipeline:
        weights = {"test": "fixture"}

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured["load"] = kwargs
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def plan(self, **kwargs):
            captured["plan"] = kwargs
            return SimpleNamespace(request="plan", truncated=False, abc="X:1\nK:C\nCDEF|")

        def generate_semantic(self, plan, **kwargs):
            captured["sampling"] = kwargs["sampling"]
            return SimpleNamespace(truncated=True)

        def synthesize(self, semantic):
            return "latents"

        def decode(self, latents):
            assert latents == "latents"
            return np.ones((480001, 2), dtype=np.float32) * .1

        def effective_config(self, request, **kwargs):
            return kwargs

    monkeypatch.setitem(sys.modules, "yue2", SimpleNamespace(YuE2Pipeline=Pipeline))
    def write_audio(path, audio, rate, subtype):
        assert subtype == "PCM_16"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((2, 2, rate, 0, "NONE", "not compressed"))
            wav.writeframes((audio * 32767).astype("<i2").tobytes())
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(write=write_audio))
    request = MusicRequest(prompt="中文钢琴", duration_seconds=10, cot="melody", abc="X:1\nK:C\nCDEF|", cfg_scale=1.2)
    source = tmp_path / "request.json"
    source.write_text(request.model_dump_json(), encoding="utf-8")
    output = tmp_path / "audio.wav"
    generate(SimpleNamespace(request=source, output=output, progress=tmp_path / "progress.json",
                             model_dir=tmp_path, device="cuda", memory_budget_gib=12, offload_ar=True))
    assert captured["load"]["local_files_only"] is True
    assert captured["load"]["backend"] == "torch-eager"
    assert captured["sampling"] == {"max_tokens": 250, "min_tokens": 200}
    assert captured["plan"]["abc"] == request.abc
    assert captured["plan"]["lyrics"] == ""
    assert "instrumental, no vocals" in captured["plan"]["style"]
    assert captured["plan"]["cfg_scale"] == 1.2
    assert MusicRunner._validate_audio(output, 10) == 10
    result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert result["truncated"] == {"abc": False, "semantic": True, "audio_cropped": True}
