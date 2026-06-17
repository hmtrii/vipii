"""Recognizer registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from vipii.recognizers.base import Recognizer
from vipii.recognizers.pattern import PatternRecognizer


def built_in_recognizers() -> list[PatternRecognizer]:
    from vipii.config import load_builtin_recognizers

    return load_builtin_recognizers()


@dataclass
class RecognizerRegistry:
    recognizers: list[Recognizer] = field(default_factory=built_in_recognizers)

    def add(self, recognizer: Recognizer) -> None:
        self.recognizers.append(recognizer)

    def __iter__(self) -> Iterable[Recognizer]:
        return iter(self.recognizers)
