"""Vehicle registration plates."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_vehicle_plate


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="vehicle_plate",
        label="VEHICLE_PLATE",
        validator=valid_vehicle_plate,
        patterns=[
            Pattern(
                label="VEHICLE_PLATE",
                regex=r"\b\d{2}[A-Z][A-Z0-9]?[ -.]?\d{3,5}(?:\.\d{2})?\b",
                context_words=[
                    "biển số",
                    "bien so",
                    "biển kiểm soát",
                    "xe",
                ],
                base_score=0.45,
            ),
        ],
    )
