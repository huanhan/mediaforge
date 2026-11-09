"""Numerical and lifetime regressions for the YuE2 attention OOM fix."""
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from mediaforge_service.music_memory import bounded_sdpa, yue2_memory_scope


def test_single_token_decode_keeps_native_dispatch(monkeypatch):
    q = torch.randn(1, 4, 1, 8)
    k, v = torch.randn(2, 1, 2, 27, 8)
    original = torch.nn.functional.scaled_dot_product_attention
    expected = original(q, k, v, enable_gqa=True)
    calls = []

    def observe(query, key, value, **kwargs):
        calls.append((key is k, value is v, kwargs["enable_gqa"]))
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe)
    torch.testing.assert_close(bounded_sdpa(q, k, v), expected)
    assert calls == [(True, True, True)]


@pytest.mark.parametrize("causal,key_length,mask_kind", [
    (True, 19, None), (True, 27, None), (False, 27, None),
    (False, 27, "bool"), (False, 27, "float"), (False, 27, "broadcast"),
])
def test_tiled_gqa_matches_full_attention(monkeypatch, causal, key_length, mask_kind):
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(1, 4, 19, 8, generator=generator)
    k = torch.randn(1, 2, key_length, 8, generator=generator)
    v = torch.randn(1, 2, key_length, 8, generator=generator)
    mask = None
    if mask_kind:
        mask = torch.ones(19 if mask_kind != "broadcast" else 1, key_length, dtype=torch.bool)
        mask[..., -3:] = False
        if mask_kind == "float":
            mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, float("-inf"))
    original = torch.nn.functional.scaled_dot_product_attention
    expected = original(q, k, v, attn_mask=mask, is_causal=causal, enable_gqa=True)
    lengths = []

    def observe(query, *args, **kwargs):
        lengths.append(query.shape[-2])
        return original(query, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe)
    actual = bounded_sdpa(q, k, v, attn_mask=mask, is_causal=causal, query_rows=5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert lengths == [5, 5, 5, 4]


def test_worker_scope_preserves_mlp_nar_progress_and_restores_on_failure(monkeypatch):
    sizes, progress, upstream = [], [], []

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(8, 16)
            self.second = torch.nn.Linear(16, 8)

        def forward(self, x):
            sizes.append(x.shape[-2])
            return self.second(torch.nn.functional.silu(self.first(x)))

    def synthesize(*args, **kwargs):
        kwargs["on_progress"](3, 32)
        return "latents"

    original_sdpa, original_attention, original_mlp = object(), object(), MLP.forward
    modeling = SimpleNamespace(sdpa=original_sdpa, MLP=MLP)
    nar = SimpleNamespace(attention=original_attention, synthesize=synthesize)
    monkeypatch.setitem(sys.modules, "yue2", SimpleNamespace(modeling_yue2=modeling, nar=nar))
    module = MLP().eval()
    x = torch.randn(1, 19, 8)
    q, k, v = torch.randn(3, 19, 4, 8)
    with torch.inference_mode():
        expected = module(x)
        expected_attention = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True)
        sizes.clear()
        with pytest.raises(RuntimeError, match="test failure"):
            with yue2_memory_scope(query_rows=5, mlp_rows=7, on_progress=lambda *p: progress.append(p)):
                torch.testing.assert_close(module(x), expected)
                assert sizes == [7, 7, 5]
                actual = nar.attention(q, k, v, causal=True)
                torch.testing.assert_close(actual, expected_attention[0].transpose(0, 1))
                assert nar.synthesize(on_progress=lambda *p: upstream.append(p)) == "latents"
                raise RuntimeError("test failure")
    assert progress == upstream == [(3, 32)]
    assert modeling.sdpa is original_sdpa and nar.attention is original_attention
    assert MLP.forward is original_mlp and nar.synthesize is synthesize
