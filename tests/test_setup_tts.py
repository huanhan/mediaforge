import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from mediaforge_service import setup_tts
from mediaforge_service.config import Settings
from mediaforge_service.speech import SpeechRunner


def test_setup_requires_explicit_noncommercial_acceptance(monkeypatch, tmp_path):
    downloads = []
    monkeypatch.setattr(setup_tts, "download_models", lambda root: downloads.append(root))
    with pytest.raises(SystemExit):
        setup_tts.main(["--models-only", "--model-dir", str(tmp_path)])
    assert not downloads
    setup_tts.main(["--models-only", "--model-dir", str(tmp_path), "--accept-non-commercial-license"])
    assert downloads == [tmp_path]


def test_manifest_pins_code_weights_and_model_license():
    manifest = setup_tts.MANIFEST
    assert len(manifest["source_revision"]) == 40
    for model in manifest["models"]:
        assert len(model["revision"]) == 40
        assert any(item["name"] == "LICENSE" for item in model["files"])
        assert all(len(item["sha256"]) == 64 and item["size"] > 0 for item in model["files"])


def test_runtime_uses_managed_python_and_never_launches_api(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(setup_tts.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(setup_tts, "install_source", lambda _: None)
    monkeypatch.setattr(setup_tts.subprocess, "run", lambda command, **_: commands.append(command))
    monkeypatch.setattr(setup_tts.subprocess, "check_output", lambda *_, **__: str(tmp_path))
    setup_tts.main(["--runtime-only", "--runtime-dir", str(tmp_path / "env"),
                    "--source-dir", str(tmp_path / "src"), "--accept-non-commercial-license"])
    assert "--managed-python" in commands[0] and "--allow-existing" in commands[0]
    assert not any("uvicorn" in command or "serve" in command for command in commands)
    assert (tmp_path / "breeze_source.pth").read_text().strip() == str(tmp_path / "src")


def test_readiness_requires_acceptance_and_only_probes_imports(monkeypatch, tmp_path):
    from mediaforge_service import speech
    interpreter = tmp_path / "python.exe"
    interpreter.touch()
    monkeypatch.setattr(speech, "missing_model_files", lambda _: [])
    monkeypatch.setattr(speech.shutil, "which", lambda _: "ffmpeg")
    probes = []
    def run(command, **kwargs):
        probes.append(command)
        return SimpleNamespace(returncode=0, stdout='{"status":"ready"}', stderr="")
    monkeypatch.setattr(speech.subprocess, "run", run)
    runner = SpeechRunner(Settings(tts_python=interpreter, tts_model_path=tmp_path), lambda _: Path())
    assert runner.readiness()["status"] == "not_ready" and not probes
    model = tmp_path / setup_tts.model_name()
    model.mkdir()
    (model / "H3_LICENSE_ACCEPTANCE.json").write_text(json.dumps({"accepted": True, "revision": setup_tts.MANIFEST["models"][0]["revision"]}))
    result = runner.readiness()
    assert result["status"] == "ready" and result["commercial_use"] is False
    assert "--check" in probes[0] and "--request" not in probes[0]
