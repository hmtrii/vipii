"""Số BHXH (social insurance number)."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_social_insurance


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="social_insurance",
        label="SOCIAL_INSURANCE_NUMBER",
        validator=valid_social_insurance,
        token_window=1,
        patterns=[
            Pattern(
                label="SOCIAL_INSURANCE_NUMBER",
                regex=r"(?<!\d)\d{10}(?!\d)",
                context_words=[
                    "bhxh",
                    "bảo hiểm xã hội",
                    "bao hiem xa hoi",
                    "số bhxh",
                    "so bhxh",
                ],
                base_score=0.45,
            ),
        ],
    )
