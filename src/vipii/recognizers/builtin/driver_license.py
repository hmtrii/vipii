"""Giấy phép lái xe (driver license number)."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_driver_license


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="driver_license",
        label="DRIVER_LICENSE",
        validator=valid_driver_license,
        token_window=1,
        patterns=[
            Pattern(
                label="DRIVER_LICENSE",
                regex=r"(?<!\d)\d{12}(?!\d)",
                context_words=[
                    "gplx",
                    "giấy phép lái xe",
                    "giay phep lai xe",
                    "bằng lái",
                    "bang lai",
                ],
                base_score=0.35,
            ),
        ],
    )
