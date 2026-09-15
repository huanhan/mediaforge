from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from .backend import BackendError, OutputArtifact, StoredAsset
from .models import VIDEO_EXTENSIONS


def safe_upload_name(original_name: str | None) -> str:
    name = Path((original_name or "upload.bin").replace("\\", "/")).name
    stem = re.sub(r"[^\w.-]+", "_", Path(name).stem, flags=re.UNICODE).strip("._")[:80]
    suffix = Path(name).suffix.lower()[:16]
    return f"{stem or 'upload'}-{uuid.uuid4().hex[:12]}{suffix}"


def _inside(root: Path, candidate: Path, *, strict: bool) -> Path:
    resolved_root = root.resolve()
    try:
        resolved = candidate.resolve(strict=strict)
    except FileNotFoundError as exc:
        raise BackendError("file not found", status_code=404, code="file_not_found") from exc
    if not resolved.is_relative_to(resolved_root):
        raise BackendError("invalid file path", status_code=400, code="invalid_file_path")
    return resolved


class LocalAssetStore:
    def __init__(self, input_root: Path, output_root: Path, max_upload_bytes: int) -> None:
        self.input_root = input_root.resolve()
        self.output_root = output_root.resolve()
        self.max_upload_bytes = max_upload_bytes

    def initialize(self) -> None:
        self.input_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    async def store_upload(
        self,
        upload: UploadFile,
        *,
        kind: Literal["image", "video", "audio"],
    ) -> StoredAsset:
        subfolder = f"minimaxh3/{datetime.now(UTC):%Y%m%d}"
        filename = safe_upload_name(upload.filename)
        folder = _inside(self.input_root, self.input_root / subfolder, strict=False)
        folder.mkdir(parents=True, exist_ok=True)
        target = _inside(self.input_root, folder / filename, strict=False)

        total = 0
        try:
            with target.open("xb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_upload_bytes:
                        raise BackendError(
                            "file is too large",
                            status_code=413,
                            code="upload_too_large",
                        )
                    destination.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise

        return StoredAsset(
            file_id=f"{subfolder}/{filename}",
            filename=filename,
            subfolder=subfolder,
            kind=kind,
        )

    def resolve_input(self, file_id: str) -> Path:
        normalized = file_id.replace("\\", "/").strip("/")
        if not normalized or ":" in normalized or ".." in normalized.split("/"):
            raise BackendError("invalid file_id", status_code=400, code="invalid_file_id")
        path = _inside(self.input_root, self.input_root / normalized, strict=True)
        if not path.is_file():
            raise BackendError("input file not found", status_code=404, code="file_not_found")
        return path

    def output_path(self, task_id: str, filename_prefix: str | None = None) -> OutputArtifact:
        prefix = (filename_prefix or "minimax_h3/api").replace("\\", "/").strip("/")
        subfolder = str(Path(prefix).parent).replace("\\", "/")
        if subfolder == ".":
            subfolder = ""
        stem = Path(prefix).name
        filename = f"{stem}_{task_id[:12]}.mp4"
        folder = _inside(self.output_root, self.output_root / subfolder, strict=False)
        folder.mkdir(parents=True, exist_ok=True)
        path = _inside(self.output_root, folder / filename, strict=False)
        return OutputArtifact(filename=filename, subfolder=subfolder, path=path)

    def music_output_path(self, task_id: str) -> OutputArtifact:
        video = self.output_path(task_id, "music/song")
        return OutputArtifact(filename=video.path.with_suffix(".wav").name,
                              subfolder=video.subfolder, path=video.path.with_suffix(".wav"))

    def upscale_output_path(
        self,
        task_id: str,
        filename_prefix: str | None = None,
        source_file_id: str | None = None,
    ) -> OutputArtifact:
        prefix = (filename_prefix or "upscale/api").replace("\\", "/").strip("/")
        subfolder = str(Path(prefix).parent).replace("\\", "/")
        if subfolder == ".":
            subfolder = ""
        stem = Path(prefix).name
        suffix = (
            ".mp4"
            if Path(source_file_id or "").suffix.lower() in VIDEO_EXTENSIONS
            else ".png"
        )
        filename = f"{stem}_{task_id[:12]}{suffix}"
        folder = _inside(self.output_root, self.output_root / subfolder, strict=False)
        folder.mkdir(parents=True, exist_ok=True)
        path = _inside(self.output_root, folder / filename, strict=False)
        return OutputArtifact(filename=filename, subfolder=subfolder, path=path)

    def resolve_file(
        self,
        *,
        filename: str,
        subfolder: str,
        file_type: Literal["input", "output", "temp"],
    ) -> Path:
        normalized_filename = filename.replace("\\", "/")
        normalized_subfolder = subfolder.replace("\\", "/").strip("/")
        if (
            "/" in normalized_filename
            or normalized_filename in {".", ".."}
            or "\x00" in normalized_filename
            or "\r" in normalized_filename
            or "\n" in normalized_filename
            or ":" in normalized_subfolder
            or ".." in normalized_subfolder.split("/")
            or "\x00" in normalized_subfolder
        ):
            raise BackendError("invalid file path", status_code=400, code="invalid_file_path")
        if file_type == "temp":
            raise BackendError("temporary files are not exposed", status_code=404, code="file_not_found")
        root = self.input_root if file_type == "input" else self.output_root
        path = _inside(root, root / normalized_subfolder / normalized_filename, strict=True)
        if not path.is_file():
            raise BackendError("file not found", status_code=404, code="file_not_found")
        return path
