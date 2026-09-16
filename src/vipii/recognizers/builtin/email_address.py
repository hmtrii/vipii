"""Email addresses."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_email


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="email_address",
        label="EMAIL_ADDRESS",
        validator=valid_email,
        patterns=[
            Pattern(
                label="EMAIL_ADDRESS",
                regex=r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                context_words=[
                    "email",
                    "e-mail",
                    "thư điện tử",
                    "thu dien tu",
                    "mail",
                ],
                base_score=0.75,
            ),
        ],
    )
