"""Install the pinned, native Transformers Qwen3-ASR-1.7B checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
MODEL_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
REQUIRED_FILES = ("config.json", "generation_config.json", "model.safetensors",
                  "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/asr/Qwen3-ASR-1.7B"))
    args = parser.parse_args(argv)
    root = args.output.expanduser().resolve()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(root),
                      allow_patterns=[*REQUIRED_FILES, "README.md"], max_workers=2)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file() or not (root / name).stat().st_size]
    if missing:
        raise RuntimeError("Incomplete ASR model: " + ", ".join(missing))
    (root / "REVISION").write_text(MODEL_REVISION + "\n", encoding="utf-8")
    print(f"Installed {MODEL_ID} at {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
