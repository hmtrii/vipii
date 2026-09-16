"""IPv4 addresses."""

from __future__ import annotations

from vipii.models import Pattern
from vipii.recognizers.pattern import PatternRecognizer
from vipii.recognizers.validators import valid_ip_address


def recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="ip_address",
        label="IP_ADDRESS",
        validator=valid_ip_address,
        patterns=[
            Pattern(
                label="IP_ADDRESS",
                regex=r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b",
                context_words=[
                    "ip",
                    "địa chỉ ip",
                    "dia chi ip",
                    "thiết bị",
                    "thiet bi",
                    "network",
                ],
                base_score=0.65,
            ),
        ],
    )
