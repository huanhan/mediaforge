"""Memory-bounded attention for the pinned YuE2 runtime in its isolated worker.

Windows PyTorch lacks Flash Attention. Native GQA then falls back to math SDPA,
materializing quadratic FP32 attention scores. Expanding only K/V heads allows
the built-in efficient CUDA kernel to run; unsupported inputs use query tiles.
All queries retain their original keys, masks and positions. No song is split.
"""
from __future__ import annotations

from contextlib import contextmanager


def bounded_sdpa(query, key, value, *, attn_mask=None, is_causal=False, query_rows=128):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    grouped = query.shape[1] != key.shape[1]
    if query.shape[-2] == 1:
        # AR token decoding is already bounded (one score row). Keep its native
        # dispatch rather than probing kernels and copying K/V on every token.
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
                                             is_causal=is_causal, enable_gqa=grouped)
    if query.device.type == "cuda":
        params = torch.backends.cuda.SDPAParams(query, key, value, attn_mask, 0., is_causal, grouped)
        if torch.backends.cuda.can_use_flash_attention(params):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
                                                     is_causal=is_causal, enable_gqa=grouped)
        if grouped:
            groups = query.shape[1] // key.shape[1]
            key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
            grouped = False
        params = torch.backends.cuda.SDPAParams(query, key, value, attn_mask, 0., is_causal, False)
        if torch.backends.cuda.can_use_efficient_attention(params):
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
                                                     is_causal=is_causal)

    # Bound math fallback, including CPU. Rectangular causal SDPA is upper-left
    # aligned; later tiles need absolute query positions, not a fresh triangle.
    result = torch.empty_like(query)
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, query.shape[-2], query_rows):
            end = min(start + query_rows, query.shape[-2])
            mask = attn_mask
            if mask is not None and mask.shape[-2] != 1:
                mask = mask[..., start:end, :]
            causal = is_causal and start == 0
            if is_causal and start:
                visible = (torch.arange(key.shape[-2], device=query.device)[None, :] <=
                           torch.arange(start, end, device=query.device)[:, None])
                if mask is None:
                    mask = visible
                elif mask.dtype == torch.bool:
                    mask = mask & visible
                else:
                    mask = mask.masked_fill(~visible, float("-inf"))
            result[..., start:end, :] = F.scaled_dot_product_attention(
                query[..., start:end, :], key, value, attn_mask=mask,
                is_causal=causal, enable_gqa=grouped)
    return result


@contextmanager
def yue2_memory_scope(*, query_rows=128, mlp_rows=256, on_progress=None):
    """Worker-local hooks; restore even on OOM. Never alter installed packages."""
    from yue2 import modeling_yue2, nar

    original_sdpa = modeling_yue2.sdpa
    original_attention = nar.attention
    original_mlp = modeling_yue2.MLP.forward
    original_synthesize = nar.synthesize

    def sdpa(query, key, value, *, attn_mask=None, is_causal=False):
        return bounded_sdpa(query, key, value, attn_mask=attn_mask,
                            is_causal=is_causal, query_rows=query_rows)

    def attention(q, k, v, *, causal=False, backend="sdpa", query_chunk_size=None):
        if backend != "sdpa":
            return original_attention(q, k, v, causal=causal, backend=backend,
                                      query_chunk_size=query_chunk_size or query_rows)
        result = bounded_sdpa(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                              v.transpose(0, 1)[None], is_causal=causal,
                              query_rows=query_chunk_size or query_rows)
        return result[0].transpose(0, 1)

    def mlp(module, x):
        if x.shape[-2] <= mlp_rows:
            return original_mlp(module, x)
        import torch
        result = torch.empty_like(x)
        for start in range(0, x.shape[-2], mlp_rows):
            result[..., start:start + mlp_rows, :] = original_mlp(module, x[..., start:start + mlp_rows, :])
        return result

    def synthesize(*args, **kwargs):
        upstream_progress = kwargs.get("on_progress")

        def progress(completed, total):
            if upstream_progress is not None:
                upstream_progress(completed, total)
            if on_progress is not None:
                on_progress(completed, total)

        kwargs["on_progress"] = progress
        return original_synthesize(*args, **kwargs)

    try:
        modeling_yue2.sdpa, nar.attention = sdpa, attention
        modeling_yue2.MLP.forward, nar.synthesize = mlp, synthesize
        yield dict(version=1, attention="flash_or_efficient_or_tiled_math",
                   query_rows=query_rows, mlp_rows=mlp_rows)
    finally:
        modeling_yue2.sdpa, nar.attention = original_sdpa, original_attention
        modeling_yue2.MLP.forward, nar.synthesize = original_mlp, original_synthesize
