from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from mediaforge_service.config import Settings
from mediaforge_service.memory import (
    GenerationMemoryError, check_allocation, chunked_linear, chunked_mlp,
    chunked_rope, h3_memory_scope, mlp_scope,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [2048, 4096])
def test_streaming_mlp_equivalence_and_bounded_intermediate(device, rows):
    torch = pytest.importorskip("torch")
    eager = pytest.importorskip("comfy_kitchen.backends.eager")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(4099, 128, device=device, dtype=torch.bfloat16)
    w1 = torch.randint(-50, 50, (512, 128), device=device, dtype=torch.int8)
    w2 = torch.randint(-50, 50, (128, 256), device=device, dtype=torch.int8)
    scale = torch.tensor(0.01, device=device)
    seen = []
    def original(value):
        intermediate = eager.int8_linear(value, w1, scale)
        seen.append(intermediate.shape[0])
        return eager.int8_linear(intermediate, w2, scale, input_act="swiglu")
    expected = original(x)
    seen.clear()
    actual = chunked_mlp(original, rows)(x)
    assert seen == ([2048, 2048, 3] if rows == 2048 else [4096, 3])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("override", [False, True])
def test_mlp_scope_restores_on_cancellation(override):
    torch = pytest.importorskip("torch")
    module = torch.nn.Identity()
    if override:
        module.forward = lambda x: x + 1
    before = dict(module.__dict__)
    canceled = Mock(side_effect=RuntimeError("canceled"))
    with pytest.raises(RuntimeError, match="canceled"):
        with mlp_scope([module], 2, lambda x: True, None, canceled):
            module(torch.zeros(3, 4))
    assert module.__dict__ == before


def test_mlp_zero_leaves_forward_untouched():
    module = SimpleNamespace(forward=lambda x: x)
    original = module.forward
    with mlp_scope([module], 0, Mock(), Mock(), Mock()):
        assert module.forward is original
        assert module.forward(42) == 42
    assert module.forward is original


@pytest.mark.parametrize("free, fraction, required, allowed", [
    (3, 0.7, 3.6, True),  # Cached blocks are reclaimable; free alone is not capacity.
    (3, 0.4, 3.6, False),  # Allocator budget is the limiting factor.
    (0.5, 1.0, 4, False),  # External GPU allocations limit even a generous budget.
])
def test_allocation_guard(free, fraction, required, allowed):
    gib = 2**30
    cuda = SimpleNamespace(
        mem_get_info=Mock(return_value=(free * gib, 12 * gib)),
        memory_allocated=Mock(return_value=2.5 * gib),
        memory_reserved=Mock(return_value=5.5 * gib),
        get_per_process_memory_fraction=Mock(return_value=fraction),
    )
    fake = SimpleNamespace(cuda=cuda)
    if allowed:
        check_allocation(fake, "cuda:0", required * gib)
    else:
        with pytest.raises(GenerationMemoryError, match="显存容量保护"):
            check_allocation(fake, "cuda:0", required * gib)


@pytest.mark.parametrize("values", [{"eager_chunk_rows": -1}, {"mlp_chunk_rows": -1}, {"eager_memory_fraction": 0},
                                    {"eager_memory_fraction": 1.1}, {"eager_memory_fraction": float("nan")}])
def test_invalid_policy(values):
    with pytest.raises(ValueError):
        Settings(**values)


def test_env_policy(monkeypatch):
    assert Settings().mlp_chunk_rows == 0
    monkeypatch.delenv("H3_MLP_CHUNK_ROWS", raising=False)
    assert Settings.from_env().mlp_chunk_rows == 0
    monkeypatch.setenv("H3_EAGER_CHUNK_ROWS", "1024")
    monkeypatch.setenv("H3_MLP_CHUNK_ROWS", "8192")
    monkeypatch.setenv("H3_EAGER_MEMORY_FRACTION", "0.6")
    settings = Settings.from_env()
    assert (settings.eager_chunk_rows, settings.eager_memory_fraction) == (1024, 0.6)
    assert settings.mlp_chunk_rows == 8192
    monkeypatch.setenv("H3_MLP_CHUNK_ROWS", "0")
    assert Settings.from_env().mlp_chunk_rows == 0


@pytest.mark.parametrize("device, rows", [("cpu", 2048), ("cuda", 0)])
def test_disabled_scope_never_touches_cuda(device, rows):
    torch = Mock()
    with h3_memory_scope(torch, SimpleNamespace(type=device), Settings(eager_chunk_rows=rows)):
        pass
    assert not torch.mock_calls


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("activation", [None, "swiglu"])
@pytest.mark.parametrize("convrot", [False, True])
def test_linear_equivalence(device, activation, convrot):
    torch = pytest.importorskip("torch")
    eager = pytest.importorskip("comfy_kitchen.backends.eager")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    x = torch.randn(1, 4099, 256 if activation else 128, device=device, dtype=torch.bfloat16)
    w = torch.randint(-100, 100, (64, 128), device=device, dtype=torch.int8)
    scale = torch.full((64,) if convrot else (), 0.01, device=device)
    bias = torch.randn(64, device=device, dtype=torch.bfloat16)
    options = dict(input_act=activation, convrot=convrot, convrot_groupsize=64)
    expected = eager.int8_linear(x, w, scale, bias, **options)
    actual = chunked_linear(torch, eager.int8_linear, 2048)(x, w, scale, bias, **options)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rope_strided_tail_equivalence(device):
    torch = pytest.importorskip("torch")
    eager = pytest.importorskip("comfy_kitchen.backends.eager")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    packed = torch.randn(1, 4099, 6, 64, device=device, dtype=torch.bfloat16)
    expected = packed.clone()
    freqs = torch.randn(1, 4099, 1, 16, 2, 2, device=device)
    scale = torch.randn(64, device=device)
    eager.rms_rope_split_half_(expected[:, :, :2], expected[:, :, 2:4], freqs, scale, rot_dim=32)
    q, k = packed[:, :, :2], packed[:, :, 2:4]
    result = chunked_rope(eager.rms_rope_split_half_, 2048)(q, k, freqs, scale, rot_dim=32)
    assert result[0] is q and result[1] is k
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["eager", "cuda"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("mlp_rows", [0, 4096])
def test_scope_restores_state(backend, fail, mlp_rows):
    torch = pytest.importorskip("torch")
    eager = pytest.importorskip("comfy_kitchen.backends.eager")
    from comfy_kitchen.registry import registry
    original = (eager.int8_linear, eager.rms_rope_split_half_)
    cuda = SimpleNamespace(get_per_process_memory_fraction=Mock(return_value=0.9),
                           set_per_process_memory_fraction=Mock())
    fake = SimpleNamespace(cuda=cuda, bfloat16=torch.bfloat16)
    device = torch.device("cuda:0")
    with (
        patch.object(registry, "get_capable_backend", return_value=backend),
        patch("mediaforge_service.memory.mlp_scope", wraps=mlp_scope) as scoped,
    ):
        try:
            with h3_memory_scope(fake, device, Settings(mlp_chunk_rows=mlp_rows),
                                 diffusion_model=object() if mlp_rows == 0 else None):
                assert eager.int8_linear is not original[0]
                # CPU and unrelated calls bypass the CUDA-only adapter.
                x = torch.zeros(2, 128, dtype=torch.bfloat16)
                w = torch.zeros(32, 128, dtype=torch.int8)
                eager.int8_linear(x, w, torch.tensor(1.0))
                if fail:
                    raise RuntimeError("canceled or failed")
        except RuntimeError:
            assert fail
    assert scoped.call_args.args[1] == mlp_rows
    assert (eager.int8_linear, eager.rms_rope_split_half_) == original
    assert cuda.set_per_process_memory_fraction.call_count == (2 if backend == "eager" else 0)
    if backend == "eager":
        assert cuda.set_per_process_memory_fraction.call_args.args == (0.9, device)
