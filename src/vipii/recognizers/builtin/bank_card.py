"""Payment card numbers, including NAPAS."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_bank_card


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="bank_card",
        label="BANK_CARD",
        validator=valid_bank_card,
        patterns=[
            Pattern(
                label="BANK_CARD",
                regex=r"(?<!\d)\d(?:[ -]?\d){15,18}(?!\d)",
                context_words=[
                    "thẻ",
                    "the",
                    "card",
                    "napas",
                    "visa",
                    "mastercard",
                ],
                base_score=0.35,
            ),
        ],
    )
