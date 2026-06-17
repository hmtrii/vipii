"""Regex-based recognizer implementation."""

from __future__ import annotations

from dataclasses import dataclass

from vipii.models import Pattern, PIIMatch
from vipii.recognizers.validators import Validator
from vipii.scoring import score_with_context


@dataclass
class PatternRecognizer:
    """Regex recognizer with optional validation and context scoring."""

    name: str
    label: str
    patterns: list[Pattern]
    validator: Validator | None = None
    token_window: int = 8

    def recognize(self, text: str) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        for pattern in self.patterns:
            for match in pattern.compiled.finditer(text):
                value = match.group(0)
                if self.validator and not self.validator(value):
                    continue
                score = score_with_context(
                    text,
                    match.start(),
                    match.end(),
                    pattern.base_score,
                    tuple(pattern.context_words or ()),
                    token_window=self.token_window,
                )
                matches.append(
                    PIIMatch(
                        label=pattern.label,
                        start=match.start(),
                        end=match.end(),
                        text=value,
                        score=score,
                        recognizer=pattern.recognizer or self.name,
                    )
                )
        return matches


def custom_pattern_recognizer(pattern: Pattern) -> PatternRecognizer:
    pattern = Pattern(
        label=pattern.label,
        regex=pattern.regex,
        context_words=pattern.context_words,
        base_score=pattern.base_score,
        flags=pattern.flags,
        recognizer=pattern.recognizer or "custom",
    )
    return PatternRecognizer(name="custom", label=pattern.label, patterns=[pattern])
