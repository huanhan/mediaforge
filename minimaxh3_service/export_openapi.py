from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .api import app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the MiniMax H3 OpenAPI document")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/openapi.json"),
        help="destination JSON path (default: docs/openapi.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the destination does not match the generated schema",
    )
    return parser


def render_openapi() -> str:
    return json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    rendered = render_openapi()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"OpenAPI document is out of date: {output}")
            return 1
        print(f"OpenAPI document is current: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote OpenAPI document: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
