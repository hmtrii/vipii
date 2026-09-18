"""Device identifiers: UUID, MAC address and IMEI."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_device_id


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="device_id",
        label="DEVICE_ID",
        validator=valid_device_id,
        patterns=[
            Pattern(
                label="DEVICE_ID",
                regex=r"\b[0-9A-F]{8}-[0-9A-F]{4}-[1-5][0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}\b",
                context_words=[
                    "device",
                    "thiết bị",
                    "thiet bi",
                    "mã thiết bị",
                    "ma thiet bi",
                    "uuid",
                ],
                base_score=0.65,
            ),
            Pattern(
                label="DEVICE_ID",
                regex=r"\b[0-9A-F]{2}(?::[0-9A-F]{2}){5}\b",
                context_words=[
                    "mac",
                    "thiết bị",
                    "thiet bi",
                    "địa chỉ mac",
                    "dia chi mac",
                ],
                base_score=0.6,
            ),
            Pattern(
                label="DEVICE_ID",
                regex=r"(?<!\d)\d{15}(?!\d)",
                context_words=[
                    "imei",
                    "device",
                    "thiết bị",
                    "thiet bi",
                    "mã thiết bị",
                    "ma thiet bi",
                ],
                base_score=0.4,
            ),
        ],
    )
