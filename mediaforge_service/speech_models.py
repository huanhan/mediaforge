"""One speech request is one shot. Reference audio is always a local uploaded asset."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .models import validate_file_id

SPEAKERS = ("Serena", "Vivian", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee")

class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=1000)
    speaker: Literal["Serena", "Vivian", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"] = "Serena"
    model: Literal["Breeze-TTS-2"] = "Breeze-TTS-2"
    language: Literal["Auto", "Chinese", "English"] = "Chinese"
    reference_file_id: str | None = Field(default=None, min_length=1, max_length=512)
    reference_text: str | None = Field(default=None, min_length=1, max_length=1500)
    speed: float = Field(default=1, ge=0.5, le=2, allow_inf_nan=False)
    mode: Literal["clone", "direction", "design"] = "clone"
    instruction: str = Field(default="", max_length=500)
    cfg_scale: float = Field(default=4, ge=1, le=6, allow_inf_nan=False)
    pause_ms: int = Field(default=200, ge=0, le=2000)
    volume_db: float = Field(default=0, ge=-12, le=12, allow_inf_nan=False)
    temperature: float = Field(default=0.9, ge=0.1, le=1.5, allow_inf_nan=False)
    top_p: float = Field(default=1, ge=0.1, le=1, allow_inf_nan=False)
    top_k: int = Field(default=50, ge=1, le=100)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2147483647)

    @field_validator("reference_file_id")
    @classmethod
    def safe_reference(cls, value):
        return validate_file_id(value) if value is not None else value

    @model_validator(mode="after")
    def reference_pair(self):
        if bool(self.reference_text) != bool(self.reference_file_id):
            raise ValueError("Breeze cloning requires reference_file_id and the exact reference_text together")
        if self.mode in {"direction", "design"} and not self.instruction:
            raise ValueError("Voice direction/design requires an instruction")
        if self.mode == "design" and self.reference_file_id:
            raise ValueError("Voice design uses an instruction without reference audio")
        return self

class SubmittedSpeechTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int
