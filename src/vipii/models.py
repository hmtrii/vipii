"""Data models for vipii."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern as CompiledPattern


@dataclass(frozen=True)
class PIIMatch:
    """A detected PII span."""

    label: str
    start: int
    end: int
    text: str
    score: float
    recognizer: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "score": round(self.score, 4),
            "recognizer": self.recognizer,
        }


@dataclass(frozen=True)
class Pattern:
    """A user-defined or built-in regex pattern."""

    label: str
    regex: str
    context_words: tuple[str, ...] | list[str] | None = None
    base_score: float = 0.5
    flags: int = re.IGNORECASE
    recognizer: str | None = None
    compiled: CompiledPattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0 <= self.base_score <= 1:
            raise ValueError("base_score must be between 0 and 1")
        object.__setattr__(self, "compiled", re.compile(self.regex, self.flags))
        object.__setattr__(self, "context_words", tuple(self.context_words or ()))
