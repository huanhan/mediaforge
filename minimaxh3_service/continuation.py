"""Fixed latent-prefix conditioning for H3 audio/video continuation.

The predecessor stays untouched. A clean tail is copied into the new window;
Comfy's inpainting sampler re-noises the prefix at each sampling timestep
and preserves its clean value in the output using zero/one masks.
This uses native sampling masks, not TimelineDirector's dynamic drift patch.
"""
from __future__ import annotations

from typing import Any


def continue_latent(core: Any, target: dict, previous: Any, overlap_frames: int,
                    *, include_audio: bool) -> dict:
    if overlap_frames < 5 or (overlap_frames - 5) % 17:
        raise ValueError("continuation overlap must lie on the H3 17k+5 frame grid")
    video, audio = target["samples"].unbind()
    old_video, old_audio = previous.unbind()
    count = (overlap_frames - 5) // 17 * 5 + 2
    if video.shape[3:] != old_video.shape[3:]:
        raise ValueError("continuation requires matching video dimensions")
    if count >= video.shape[2] or count > old_video.shape[2]:
        raise ValueError("continuation prefix must leave room for new frames")
    video, audio = video.clone(), audio.clone()
    video[:, :, :count] = old_video[:, :, -count:].to(video)
    video_mask = core.torch.ones((1, 1, video.shape[2], 1, 1), device=video.device)
    video_mask[:, :, :count] = 0
    audio_mask = core.torch.ones((1, 1, 1, audio.shape[-1]), device=audio.device)
    if include_audio:
        audio_count = round(overlap_frames * 40 / 24)
        if audio_count >= audio.shape[-1] or audio_count > old_audio.shape[-1]:
            raise ValueError("audio continuation prefix does not fit")
        audio[..., :audio_count] = old_audio[..., -audio_count:].to(audio)
        audio_mask[..., :audio_count] = 0
    return {**target, "samples": core.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": core.nested_tensor.NestedTensor((video_mask, audio_mask))}
