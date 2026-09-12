"""Build a Windows portable ZIP in a fresh output directory."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from build_portable import build
from package_runtime import ROOT


def main() -> int:
    output = ROOT / "dist" / datetime.now().strftime("windows-%Y%m%d-%H%M%S-%f")
    print(f"Building Windows ZIP. Output: {output}", flush=True)
    try:
        build(
            output,
            runtime_source=None,
            vc_runtime=Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32",
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Build failed: {exc}", file=sys.stderr, flush=True)
        print(f"Existing artifacts are preserved in: {output}", file=sys.stderr, flush=True)
        return 1
    print(f"Windows build completed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
