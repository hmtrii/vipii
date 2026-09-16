"""Recognizer type shared by all recognizer implementations."""

from __future__ import annotations

from presidio_analyzer import EntityRecognizer

# vipii recognizers are Presidio recognizers; there is no adapter layer.
Recognizer = EntityRecognizer

__all__ = ["Recognizer"]
