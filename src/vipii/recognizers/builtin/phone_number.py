"""Vietnamese mobile and landline numbers."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_phone


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="phone_number",
        label="PHONE_NUMBER",
        validator=valid_phone,
        patterns=[
            Pattern(
                label="PHONE_NUMBER",
                regex=r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){9}(?!\d)",
                context_words=[
                    "số điện thoại",
                    "so dien thoai",
                    "điện thoại",
                    "phone",
                    "zalo",
                ],
                base_score=0.55,
            ),
        ],
    )
