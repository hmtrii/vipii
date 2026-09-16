"""Bank account numbers."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_bank_account


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="bank_account",
        label="BANK_ACCOUNT",
        validator=valid_bank_account,
        patterns=[
            Pattern(
                label="BANK_ACCOUNT",
                regex=r"(?<!\d)\d{8,16}(?!\d)",
                context_words=[
                    "tài khoản",
                    "tai khoan",
                    "stk",
                    "ngân hàng",
                    "ngan hang",
                ],
                base_score=0.35,
            ),
        ],
    )
