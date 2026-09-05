import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_runtime", ROOT / "scripts" / "package_runtime.py")
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


@pytest.mark.parametrize("runtime,path", [
    ("ComfyUI", "models/diffusion_models/private.safetensors"),
    ("ComfyUI", "input/private.png"),
    ("ComfyUI", "output/result.mp4"),
    ("ComfyUI", ".git/config"),
    ("ComfyUI", "comfy/test.safetensors"),
    ("ComfyUI", "comfy/__pycache__/utils.pyc"),
    ("SeedVR2", "example_workflows/example.json"),
    ("SeedVR2", "src/private.pt"),
    ("SeedVR2", "models/seedvr2.safetensors"),
])
def test_runtime_export_excludes_weights_and_local_data(runtime, path):
    assert not packaging.include_runtime_file(runtime, path)


@pytest.mark.parametrize("runtime,path", [
    ("ComfyUI", "comfy/ldm/minimax/model.py"),
    ("ComfyUI", "comfy/text_encoders/qwen25_tokenizer/vocab.json"),
    ("ComfyUI", "comfy/ldm/models/diffusion/ddim.py"),
    ("ComfyUI", "folder_paths.py"),
    ("ComfyUI", "LICENSE"),
    ("SeedVR2", "src/models/dit/model.py"),
    ("SeedVR2", "configs_3b/main.yaml"),
    ("SeedVR2", "pos_emb.pt"),
    ("SeedVR2", "neg_emb.pt"),
    ("SeedVR2", "inference_cli.py"),
])
def test_runtime_export_keeps_code_and_required_static_resources(runtime, path):
    assert packaging.include_runtime_file(runtime, path)
