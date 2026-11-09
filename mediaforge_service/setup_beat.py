"""Download the MIT-licensed Beat This! final0 checkpoint; never during an API request."""
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/beat/final0.ckpt"))
    args = parser.parse_args(argv)
    destination = args.output.expanduser().resolve()
    if destination.is_file():
        print(f"Already installed: {destination}")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        with urllib.request.urlopen("https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt", timeout=60) as response:
            with temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        # Validate before installation; an HTML error page must not appear ready.
        from beat_this.inference import load_model
        load_model(str(temporary), device="cpu")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Installed: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
