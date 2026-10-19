"""Forced lyric alignment in the shared GPU queue."""
import sys

from .setup_asr import ALIGNER_FILES, ALIGNER_ID
from .transcription import TranscriptionRunner


class AlignmentRunner(TranscriptionRunner):
    def readiness(self):
        result = super().readiness()
        root = self.settings.aligner_model_path.expanduser().resolve()
        missing = [name for name in ALIGNER_FILES if not (root / name).is_file() or not (root / name).stat().st_size]
        result.update(status="not_ready" if missing or result["missing_dependencies"] else "ready",
                      backend="qwen3_forced_aligner", model=ALIGNER_ID, model_path=str(root), missing_models=missing,
                      max_duration_seconds=self.max_seconds or None, word_timestamps=True, lyrics_alignment=True)
        return result

    def worker_arguments(self, request, pcm, partial, extra):
        extra.write_text(request.model_dump_json(), encoding="utf-8")
        return [sys.executable, "-m", "minimaxh3_service.alignment_worker", "--pcm", str(pcm),
                "--output", str(partial), "--request", str(extra),
                "--model", str(self.settings.aligner_model_path.expanduser().resolve()),
                "--asr-model", str(self.settings.asr_model_path.expanduser().resolve()),
                "--device", self.settings.asr_device, "--threads", str(self.settings.asr_cpu_threads)]

    def validate_document(self, document, request):
        return (document.get("schema_version") == 1 and document.get("source_file_id") == request.file_id
                and document.get("text") == request.lyrics and document.get("timing") == "forced_alignment"
                and isinstance(document.get("cues"), list) and bool(document["cues"]))
