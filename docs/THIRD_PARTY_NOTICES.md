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
In Docker, the corresponding directory is /opt/minimaxh3.

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
are recorded in minimaxh3_service/pose_models.json. These optional models are
not bundled with the application. Setup and API details are in SKELETON.md.

## Optional IndexTTS 2.5 speech

The optional installer downloads pinned IndexTTS source and weights. Their
bilibili Model Use License Agreement is retained in the downloaded source;
see https://github.com/index-tts/index-tts and index_tts_models.json for versions.
The application bundles nine synthetic voice-reference WAVs generated with
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (Apache-2.0). They are not user recordings.
Source revision and file hashes are in minimaxh3_service/voice_presets/manifest.json.
These fixed samples preserve preset IDs; synthesis uses only IndexTTS 2.5.
