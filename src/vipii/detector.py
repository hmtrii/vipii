"""Detector orchestration and redaction helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vipii.models import Pattern, PIIMatch
from vipii.recognizers import (
    NERRecognizer,
    Recognizer,
    RecognizerRegistry,
    built_in_recognizers,
    custom_pattern_recognizer,
)


class PIIDetector:
    """Detect Vietnamese structured PII with built-in and custom recognizers."""

    def __init__(
        self,
        recognizers: list[Recognizer] | None = None,
        *,
        config_path: str | Path | None = None,
        include_builtins: bool = True,
        ner_model: str | None = None,
        ner_min_score: float = 0.5,
        max_workers: int | None = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if recognizers is None:
            recognizers = built_in_recognizers() if include_builtins else []
        if config_path:
            from vipii.config import load_recognizers_from_yaml

            recognizers = [*recognizers, *load_recognizers_from_yaml(config_path)]
        if ner_model:
            recognizers = [
                *recognizers,
                NERRecognizer(model_name=ner_model, min_score=ner_min_score),
            ]
        self.registry = RecognizerRegistry(recognizers=recognizers)
        self.max_workers = max_workers

    @classmethod
    def from_yaml(cls, path: str | Path, *, include_builtins: bool = True) -> PIIDetector:
        return cls(config_path=path, include_builtins=include_builtins)

    def add_pattern(self, pattern: Pattern) -> None:
        self.registry.add(custom_pattern_recognizer(pattern))

    def add_recognizer(self, recognizer: Recognizer) -> None:
        self.registry.add(recognizer)

    def add_ner_model(self, model_name: str, *, min_score: float = 0.5) -> None:
        self.registry.add(NERRecognizer(model_name=model_name, min_score=min_score))

    def detect(self, text: str) -> list[PIIMatch]:
        recognizers = list(self.registry)
        if not recognizers:
            return []
        if len(recognizers) == 1 or self.max_workers == 1:
            candidates = recognize_sequentially(recognizers, text)
        else:
            candidates = recognize_concurrently(recognizers, text, max_workers=self.max_workers)
        return resolve_overlaps(candidates)

    def redact(self, text: str, mask: str = "*") -> str:
        matches = self.detect(text)
        redacted = []
        cursor = 0
        for match in matches:
            redacted.append(text[cursor : match.start])
            redacted.append(mask * max(1, match.end - match.start))
            cursor = match.end
        redacted.append(text[cursor:])
        return "".join(redacted)


def recognize_sequentially(recognizers: list[Recognizer], text: str) -> list[PIIMatch]:
    candidates: list[PIIMatch] = []
    for recognizer in recognizers:
        candidates.extend(recognizer.recognize(text))
    return candidates


def recognize_concurrently(
    recognizers: list[Recognizer], text: str, *, max_workers: int | None = None
) -> list[PIIMatch]:
    candidates: list[PIIMatch] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for matches in executor.map(lambda recognizer: recognizer.recognize(text), recognizers):
            candidates.extend(matches)
    return candidates


def resolve_overlaps(matches: list[PIIMatch]) -> list[PIIMatch]:
    ordered = sorted(
        matches, key=lambda item: (item.start, -item.score, -(item.end - item.start), item.label)
    )
    kept: list[PIIMatch] = []

    for candidate in ordered:
        overlaps = [match for match in kept if spans_overlap(candidate, match)]
        if not overlaps:
            kept.append(candidate)
            continue

        best_existing = max(
            overlaps, key=lambda item: (item.score, item.end - item.start, item.label)
        )
        if (candidate.score, candidate.end - candidate.start, candidate.label) > (
            best_existing.score,
            best_existing.end - best_existing.start,
            best_existing.label,
        ):
            kept = [match for match in kept if not spans_overlap(candidate, match)]
            kept.append(candidate)

    return sorted(kept, key=lambda item: (item.start, item.end, item.label))


def spans_overlap(left: PIIMatch, right: PIIMatch) -> bool:
    return left.start < right.end and right.start < left.end
