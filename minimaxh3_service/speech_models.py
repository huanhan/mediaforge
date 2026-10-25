"""One speech request is one shot. Reference audio is always a local uploaded asset."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .models import validate_file_id

SPEAKERS = ("Serena", "Vivian", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee")

class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=1000)
    speaker: Literal["Serena", "Vivian", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"] = "Serena"
    model: Literal["IndexTTS-2.5"] = "IndexTTS-2.5"
    language: Literal["Auto", "Chinese", "English", "Japanese", "Spanish", "Arabic"] = "Chinese"
    reference_file_id: str | None = Field(default=None, min_length=1, max_length=512)
    reference_text: str | None = Field(default=None, min_length=1, max_length=1500)
    speed: float = Field(default=1, ge=0.5, le=2, allow_inf_nan=False)
    emotion_mode: Literal["reference", "text", "vector"] = "reference"
    emotion_text: str = Field(default="自然、平静地讲述。", max_length=500)
    emotion_strength: float = Field(default=0.65, ge=0, le=1, allow_inf_nan=False)
    emotion_vector: list[float] = Field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0.5], min_length=8, max_length=8)
    emotion_reference_file_id: str | None = Field(default=None, min_length=1, max_length=512)
    pause_ms: int = Field(default=200, ge=0, le=2000)
    volume_db: float = Field(default=0, ge=-12, le=12, allow_inf_nan=False)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5, allow_inf_nan=False)
    top_p: float = Field(default=0.8, ge=0.1, le=1, allow_inf_nan=False)
    top_k: int = Field(default=30, ge=1, le=100)
    repetition_penalty: float = Field(default=10, ge=1, le=20, allow_inf_nan=False)
    max_segment_tokens: int = Field(default=120, ge=40, le=200)
    text_normalization: bool = True
    seed: int = Field(default=42, ge=0, le=2147483647)

    @field_validator("reference_file_id", "emotion_reference_file_id")
    @classmethod
    def safe_reference(cls, value):
        return validate_file_id(value) if value is not None else value

    @field_validator("emotion_vector")
    @classmethod
    def valid_emotions(cls, value):
        import math
        if any(not math.isfinite(n) or n < 0 or n > 1 for n in value):
            raise ValueError("Emotion values must be finite numbers between 0 and 1")
        return value

    @model_validator(mode="after")
    def reference_pair(self):
        if self.reference_text and not self.reference_file_id:
            raise ValueError("reference_text requires reference_file_id")
        if self.emotion_mode == "text" and not self.emotion_text:
            raise ValueError("Text emotion mode requires an emotion description")
        return self

class SubmittedSpeechTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int
