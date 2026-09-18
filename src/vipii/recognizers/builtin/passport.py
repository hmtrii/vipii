"""Vietnamese passport numbers."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_passport


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="passport",
        label="PASSPORT",
        validator=valid_passport,
        patterns=[
            Pattern(
                label="PASSPORT",
                regex=r"\b[A-Z][0-9]{7,8}\b",
                context_words=[
                    "hộ chiếu",
                    "ho chieu",
                    "passport",
                ],
                base_score=0.55,
            ),
        ],
    )
