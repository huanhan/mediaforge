"""Scoped H3 eager-kernel working-set limits (the GPU queue is serial)."""
from contextlib import contextmanager
from functools import wraps
import logging
import threading


_lock = threading.RLock()
logger = logging.getLogger(__name__)


class GenerationMemoryError(RuntimeError):
    """A generation exceeds the current device's working-set capacity."""


def check_allocation(torch, device, required_bytes):
    """Reject clearly impossible large outputs; this is not a full peak predictor.

    Cached allocator blocks may be reclaimed. External allocations, fragmentation,
    and later attention/VAE work can still cause OOM after this instantaneous check.
    """
    if required_bytes < 64 * 1024**2:
        return
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    cached = max(0, torch.cuda.memory_reserved(device) - allocated)
    budget = total * torch.cuda.get_per_process_memory_fraction(device)
    available = max(0, min(budget - allocated, free + cached) - 256 * 1024**2)
    if required_bytes > available:
        raise GenerationMemoryError(
            f"显存容量保护：当前张量需要 {required_bytes / 2**30:.2f} GiB，"
            f"扣除余量后可用约 {available / 2**30:.2f} GiB。"
            "请缩短视频、降低分辨率、减少参考素材，或关闭其他 GPU 程序后重试。"
            "程序未自动降低输出规格。"
        )


def chunked_linear(torch, original, rows, active=lambda x: True, check=None):
    @wraps(original)
    def linear(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16,
               convrot=False, convrot_groupsize=256, input_act=None):
        args = (weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize, input_act)
        if not active(x) or x.numel() // x.shape[-1] <= rows:
            return original(x, *args)
        flat = x.reshape(-1, x.shape[-1])
        if check is not None:
            check(x.device, flat.shape[0] * weight.shape[0] * torch.empty((), dtype=out_dtype).element_size())
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


def chunked_mlp(original, rows, active=lambda x: True, check=None, check_canceled=lambda: None):
    """Run fc1 -> SwiGLU -> fc2 together; never materialize full-sequence fc1 output."""
    @wraps(original)
    def forward(x):
        if not active(x) or x.ndim != 2 or x.shape[0] <= rows:
            return original(x)
        if check is not None:
            check(x.device, x.numel() * x.element_size())
        result = None
        for offset in range(0, x.shape[0], rows):
            check_canceled()
            chunk = original(x[offset:offset + rows])
            if result is None:
                result = chunk.new_empty((x.shape[0], chunk.shape[-1]))
            result[offset:offset + rows] = chunk
            del chunk
        return result
    return forward


@contextmanager
def mlp_scope(modules, rows, active, check, check_canceled):
    if rows == 0:
        yield
        return
    restored = []
    try:
        for module in modules:
            restored.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = chunked_mlp(module.forward, rows, active, check, check_canceled)
        yield
    finally:
        for module, had_override, override in reversed(restored):
            if had_override:
                module.forward = override
            else:
                del module.forward


@contextmanager
def h3_memory_scope(torch, device, settings, diffusion_model=None, check_canceled=lambda: None):
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
        check = lambda target, size: check_allocation(torch, target, size)
        previous = None
        try:
            # No-argument dispatch checks availability/capabilities without running a kernel.
            fallback = registry.get_capable_backend("int8_linear") == "eager"
            if fallback and settings.eager_memory_fraction < 1.0:
                previous = torch.cuda.get_per_process_memory_fraction(device)
                torch.cuda.set_per_process_memory_fraction(
                    min(previous, settings.eager_memory_fraction), device)
            eager.int8_linear = chunked_linear(torch, original_linear, settings.eager_chunk_rows, active, check)
            eager.rms_rope_split_half_ = chunked_rope(original_rope, settings.eager_chunk_rows, active)
            logger.info("H3 eager memory policy: chunk_rows=%s, allocator_fraction=%s",
                        settings.eager_chunk_rows,
                        min(previous, settings.eager_memory_fraction) if previous is not None else "unchanged")
            modules = []
            if diffusion_model is not None and settings.mlp_chunk_rows > 0:
                from comfy.ldm.minimax.model import MLP
                modules = [module for module in diffusion_model.modules() if type(module) is MLP]
            logger.info("H3 streaming MLP policy: modules=%s, rows=%s", len(modules), settings.mlp_chunk_rows)
            with mlp_scope(modules, settings.mlp_chunk_rows, active, check, check_canceled):
                yield
        finally:
            eager.int8_linear, eager.rms_rope_split_half_ = original_linear, original_rope
            if previous is not None:
                torch.cuda.set_per_process_memory_fraction(previous, device)
