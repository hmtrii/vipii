"""Recognizer protocol shared by all recognizer implementations."""

from __future__ import annotations

from typing import Protocol

from vipii.models import PIIMatch


class Recognizer(Protocol):
    """Common interface for pattern and model-backed recognizers."""

    name: str

    def recognize(self, text: str) -> list[PIIMatch]:
        """Return PII matches for text."""
        ...
