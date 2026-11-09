# Third-party components

The distribution includes CPython, PyTorch with CUDA user-space libraries, PyAV,
the dependencies recorded in requirements-lock.txt, and pinned inference source.
Their original licenses remain applicable. Python's license is in python/LICENSE.txt;
package licenses and metadata are under python/Lib/site-packages.

ComfyUI source and its GPL license are included in runtime/ComfyUI. The source
repository is https://github.com/comfy-org/ComfyUI. SeedVR2 source and its license
are included in runtime/SeedVR2; its repository is
https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.
The precise source commits and resource hashes are in runtime/runtime-manifest.json.
In Docker, the corresponding directory is /opt/mediaforge.

SeedVR2's pos_emb.pt and neg_emb.pt (about 1.2 MB combined) are fixed prompt
embeddings from its pinned source tree, required by its inference code. They are
included as runtime resources. H3 and SeedVR2 model checkpoints, user inputs and
generated outputs are not included.

The Windows package includes app-local Microsoft Visual C++ and OpenMP runtime
DLLs. Microsoft's redistribution terms apply:
https://visualstudio.microsoft.com/license-terms/vs2022-cruntime/
CUDA libraries are supplied by the official PyTorch wheels and remain subject
to NVIDIA's applicable terms. NVIDIA kernel drivers are not included.

Model licenses apply separately to models downloaded by the user; see MODELS.md.

The optional local skeleton extractor uses DWPose ONNX weights from
https://huggingface.co/yzd-v/DWPose. Download locations and SHA-256 checksums
are recorded in mediaforge_service/pose_models.json. These optional models are
not bundled with the application. Setup and API details are in SKELETON.md.

## Optional Breeze TTS 2 speech

Breeze inference source is Apache-2.0 (docs/licenses/BREEZE_CODE_LICENSE.txt).
Model weights, derivatives and self-hosted outputs use the separate BreezeBlue
Research and Non-Commercial License 1.1 (docs/licenses/BREEZE_MODEL_LICENSE.txt).
Source and checkpoint revisions/hashes are pinned in breeze_tts_models.json.
The installer requires explicit non-commercial acceptance and preserves the
upstream LICENSE and NOTICE with the checkpoint. We do not bundle model weights.

Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial License
Agreement. Copyright (c) 2026 RESONIA, INC. All Rights Reserved.

The audio tokenizer is based on Qwen3-TTS by the Alibaba Qwen Team (Apache-2.0).
This required codec is not a separate Qwen TTS synthesis backend.

Fixed voice reference WAVs were originally generated with Qwen3-TTS CustomVoice
(Apache-2.0). Their provenance and exact transcripts are retained in
voice_presets/manifest.json. Current synthesis uses only Breeze TTS 2.
