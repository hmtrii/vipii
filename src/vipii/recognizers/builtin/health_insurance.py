"""Số thẻ BHYT (health insurance card number)."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_health_insurance


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="health_insurance",
        label="HEALTH_INSURANCE_NUMBER",
        validator=valid_health_insurance,
        token_window=1,
        patterns=[
            Pattern(
                label="HEALTH_INSURANCE_NUMBER",
                regex=r"\b[A-Z]{2}[\s.-]?\d(?:[\s.-]?\d){12}\b",
                context_words=[
                    "bhyt",
                    "bảo hiểm y tế",
                    "bao hiem y te",
                    "thẻ bhyt",
                    "the bhyt",
                ],
                base_score=0.5,
            ),
            Pattern(
                label="HEALTH_INSURANCE_NUMBER",
                regex=r"(?<!\d)\d{10}(?!\d)",
                context_words=[
                    "bhyt",
                    "bảo hiểm y tế",
                    "bao hiem y te",
                    "thẻ bhyt",
                    "the bhyt",
                ],
                base_score=0.4,
            ),
        ],
    )
