"""Dates of birth in Vietnamese day-first formats."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_date_of_birth


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="date_of_birth",
        label="DATE_OF_BIRTH",
        validator=valid_date_of_birth,
        patterns=[
            Pattern(
                label="DATE_OF_BIRTH",
                regex=r"\b(?:0?[1-9]|[12][0-9]|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}\b",
                context_words=[
                    "ngày sinh",
                    "ngay sinh",
                    "sinh ngày",
                    "sinh ngay",
                    "dob",
                    "năm sinh",
                ],
                base_score=0.3,
            ),
        ],
    )
