"""Căn cước công dân (12-digit citizen ID)."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_cccd


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="cccd",
        label="CCCD",
        validator=valid_cccd,
        patterns=[
            Pattern(
                label="CCCD",
                regex=r"(?<!\d)\d{12}(?!\d)",
                context_words=[
                    "cccd",
                    "căn cước",
                    "can cuoc",
                    "định danh",
                    "dinh danh",
                ],
                base_score=0.55,
            ),
        ],
    )
