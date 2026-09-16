"""Chứng minh nhân dân (legacy 9-digit ID)."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_cmnd


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="cmnd",
        label="CMND",
        validator=valid_cmnd,
        patterns=[
            Pattern(
                label="CMND",
                regex=r"(?<!\d)\d{9}(?!\d)",
                context_words=[
                    "cmnd",
                    "chứng minh",
                    "chung minh",
                    "số cmnd",
                ],
                base_score=0.45,
            ),
        ],
    )
