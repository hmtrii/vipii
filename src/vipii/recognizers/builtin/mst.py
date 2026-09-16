"""Mã số thuế (tax code), with optional branch suffix."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_tax_code


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="mst",
        label="MST",
        validator=valid_tax_code,
        token_window=3,
        patterns=[
            Pattern(
                label="MST",
                regex=r"(?<!\d)\d{10}(?:-?\d{3})?(?!\d)",
                context_words=[
                    "mã số thuế",
                    "ma so thue",
                    "mst",
                    "tax",
                ],
                base_score=0.5,
            ),
        ],
    )
