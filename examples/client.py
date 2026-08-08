from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx


BASE_URL = os.getenv("H3_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("H3_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def upload(client: httpx.Client, path: Path, kind: str) -> str:
    with path.open("rb") as source:
        response = client.post(
            "/v1/files",
            headers=HEADERS,
            data={"kind": kind},
            files={"file": (path.name, source)},
        )
    response.raise_for_status()
    return response.json()["file_id"]


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=7200) as client:
        response = client.post(
            "/v1/generations",
            headers=HEADERS,
            json={
                "mode": "t2v",
                "prompt": "Cinematic rain at night, slow dolly forward. Audio: rain and distant traffic.",
                "duration_seconds": 4,
                "width": 608,
                "height": 352,
                "seed": 42,
            },
        )
        response.raise_for_status()
        task_id = response.json()["id"]
        print(f"task: {task_id}")

        while True:
            task = client.get(f"/v1/tasks/{task_id}", headers=HEADERS)
            task.raise_for_status()
            payload = task.json()
            print(payload["status"])
            if payload["status"] in {"succeeded", "failed", "canceled", "unknown"}:
                break
            time.sleep(2)

        if payload["status"] == "succeeded" and payload["outputs"]:
            video = client.get(payload["outputs"][0]["download_url"], headers=HEADERS)
            video.raise_for_status()
            Path("result.mp4").write_bytes(video.content)
            print("saved: result.mp4")
            return
        print(payload.get("error") or f"generation ended with {payload['status']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
