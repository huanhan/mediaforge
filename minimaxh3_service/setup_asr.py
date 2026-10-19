"""Install the pinned, native Transformers Qwen3-ASR-1.7B checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
MODEL_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
REQUIRED_FILES = ("config.json", "generation_config.json", "model.safetensors",
                  "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
ALIGNER_REVISION = "c07281df297b9905d24a508279258cccf987a064"
ALIGNER_FILES = tuple(name for name in REQUIRED_FILES if name != "generation_config.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligner", action="store_true", help="Install the optional lyric forced aligner")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    model_id, revision, files = (ALIGNER_ID, ALIGNER_REVISION, ALIGNER_FILES) if args.aligner else (MODEL_ID, MODEL_REVISION, REQUIRED_FILES)
    root = (args.output or Path("models/asr/Qwen3-ForcedAligner-0.6B" if args.aligner else "models/asr/Qwen3-ASR-1.7B")).expanduser().resolve()
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, revision=revision, local_dir=str(root),
                      allow_patterns=[*files, "README.md"], max_workers=2)
    missing = [name for name in files if not (root / name).is_file() or not (root / name).stat().st_size]
    if missing:
        raise RuntimeError("Incomplete ASR model: " + ", ".join(missing))
    (root / "REVISION").write_text(revision + "\n", encoding="utf-8")
    print(f"Installed {model_id} at {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
