"""One speech request is one shot. Reference audio is always a local uploaded asset."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .models import validate_file_id

SPEAKERS = ("Serena", "Vivian", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee")

class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=1000)
    speaker: Literal["Serena", "Vivian", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"] = "Serena"
    language: Literal["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"] = "Chinese"
    instruct: str = Field(default="", max_length=500)
    reference_file_id: str | None = Field(default=None, min_length=1, max_length=512)
    reference_text: str | None = Field(default=None, min_length=1, max_length=1500)
    speed: float = Field(default=1, ge=0.75, le=1.5, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2147483647)

    @field_validator("reference_file_id")
    @classmethod
    def safe_reference(cls, value):
        return validate_file_id(value) if value is not None else value

    @model_validator(mode="after")
    def reference_pair(self):
        if bool(self.reference_file_id) != bool(self.reference_text):
            raise ValueError("Cloning requires both reference_file_id and the exact reference_text")
        if self.reference_file_id and self.instruct:
            raise ValueError("The Base voice clone model follows the recording; instruct is only supported by preset voices")
        return self

class SubmittedSpeechTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int
