import json
from pathlib import Path

from mediaforge_service.export_openapi import main


def test_openapi_export_and_staleness_check(tmp_path: Path) -> None:
    output = tmp_path / "openapi.json"

    assert main(["--output", str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["openapi"] == "3.1.0"
    assert "/v1/generations" in document["paths"]
    assert main(["--output", str(output), "--check"]) == 0

    output.write_text("{}\n", encoding="utf-8")
    assert main(["--output", str(output), "--check"]) == 1
