"""Scoped H3 eager-kernel working-set limits (the GPU queue is serial)."""
from contextlib import contextmanager
from functools import wraps
import logging
import threading


_lock = threading.RLock()
logger = logging.getLogger(__name__)


def chunked_linear(torch, original, rows, active=lambda x: True):
    @wraps(original)
    def linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16,
               convrot=False, convrot_groupsize=256, input_act=None):
        args = (weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize, input_act)
        if not active(x) or x.numel() // x.shape[-1] <= rows:
            return original(x, *args)
        flat = x.reshape(-1, x.shape[-1])
        result = torch.empty((flat.shape[0], weight.shape[0]), device=x.device, dtype=out_dtype)
        for offset in range(0, flat.shape[0], rows):
            result[offset:offset + rows] = original(flat[offset:offset + rows], *args)
        return result.reshape(*x.shape[:-1], weight.shape[0])
    return linear


def chunked_rope(original, rows, active=lambda x: True):
    @wraps(original)
    def rope(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
        # H3 uses [batch, tokens, heads, channels] and token-aligned frequencies.
        if (not active(q) or q.ndim != 4 or k.ndim != 4 or freqs_cis.ndim != 6
                or q.shape[:2] != k.shape[:2] or freqs_cis.shape[1] != q.shape[1]
                or q.shape[1] <= rows):
            return original(q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim)
        for offset in range(0, q.shape[1], rows):
            original(q[:, offset:offset + rows], k[:, offset:offset + rows],
                     freqs_cis[:, offset:offset + rows], q_scale, k_scale, epsilon, rot_dim)
        return q, k
    return rope


@contextmanager
def h3_memory_scope(torch, device, settings):
    """Leave optimized dispatch intact; restore allocator and eager hooks on exit.

    Budget is process-wide, so callers must serialize GPU jobs (TaskManager does).
    Hooks additionally check the owning thread and device to avoid changing other calls.
    """
    if device.type != "cuda" or settings.eager_chunk_rows == 0:
        yield
        return
    import comfy_kitchen.backends.eager as eager
    from comfy_kitchen.registry import registry

    with _lock:
        owner = threading.get_ident()
        active = lambda x: threading.get_ident() == owner and x.device == device
        original_linear, original_rope = eager.int8_linear, eager.rms_rope_split_half_
        previous = None
        try:
            # No-argument dispatch checks availability/capabilities without running a kernel.
            fallback = registry.get_capable_backend("int8_linear") == "eager"
            if fallback and settings.eager_memory_fraction < 1.0:
                previous = torch.cuda.get_per_process_memory_fraction(device)
                torch.cuda.set_per_process_memory_fraction(
                    min(previous, settings.eager_memory_fraction), device)
            eager.int8_linear = chunked_linear(torch, original_linear, settings.eager_chunk_rows, active)
            eager.rms_rope_split_half_ = chunked_rope(original_rope, settings.eager_chunk_rows, active)
            logger.info("H3 eager memory policy: chunk_rows=%s, allocator_fraction=%s",
                        settings.eager_chunk_rows,
                        min(previous, settings.eager_memory_fraction) if previous is not None else "unchanged")
            yield
        finally:
            eager.int8_linear, eager.rms_rope_split_half_ = original_linear, original_rope
            if previous is not None:
                torch.cuda.set_per_process_memory_fraction(previous, device)
