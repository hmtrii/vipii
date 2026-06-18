"""Detector orchestration and redaction helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from vipii.models import Pattern, PIIMatch
from vipii.recognizers import (
    NERRecognizer,
    Recognizer,
    RecognizerRegistry,
    built_in_recognizers,
    custom_pattern_recognizer,
)

NERStrategy = Literal["always", "fallback", "uncovered", "chunked", "never"]
NER_STRATEGIES = {"always", "fallback", "uncovered", "chunked", "never"}
MIN_NER_CHUNK_LENGTH = 8
MAX_REDACTED_CHUNK_RATIO = 0.6


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
        ner_strategy: NERStrategy = "always",
        max_workers: int | None = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if ner_strategy not in NER_STRATEGIES:
            raise ValueError(
                "ner_strategy must be 'always', 'fallback', 'uncovered', 'chunked', or 'never'"
            )
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
        self.ner_strategy = ner_strategy
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
        if self.ner_strategy != "always":
            candidates = recognize_with_ner_strategy(
                recognizers,
                text,
                strategy=self.ner_strategy,
                max_workers=self.max_workers,
            )
            return resolve_overlaps(candidates)
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


def recognize_with_ner_strategy(
    recognizers: list[Recognizer],
    text: str,
    *,
    strategy: NERStrategy,
    max_workers: int | None = None,
) -> list[PIIMatch]:
    ner_recognizers = [recognizer for recognizer in recognizers if is_ner_recognizer(recognizer)]
    non_ner_recognizers = [
        recognizer for recognizer in recognizers if not is_ner_recognizer(recognizer)
    ]

    candidates = recognize_recognizer_group(non_ner_recognizers, text, max_workers=max_workers)
    if strategy == "never" or not ner_recognizers:
        return candidates
    if strategy == "fallback":
        if candidates:
            return candidates
        return recognize_recognizer_group(ner_recognizers, text, max_workers=max_workers)
    if strategy == "uncovered":
        return [
            *candidates,
            *recognize_uncovered_text(ner_recognizers, text, candidates, max_workers=max_workers),
        ]
    if strategy == "chunked":
        return [
            *candidates,
            *recognize_redacted_chunks(ner_recognizers, text, candidates, max_workers=max_workers),
        ]
    return candidates


def recognize_uncovered_text(
    recognizers: list[Recognizer],
    text: str,
    covered_matches: list[PIIMatch],
    *,
    max_workers: int | None = None,
) -> list[PIIMatch]:
    matches: list[PIIMatch] = []
    for offset, segment in uncovered_text_segments(text, resolve_overlaps(covered_matches)):
        segment_matches = recognize_recognizer_group(recognizers, segment, max_workers=max_workers)
        matches.extend(offset_match(match, offset, text) for match in segment_matches)
    return matches


def uncovered_text_segments(text: str, covered_matches: list[PIIMatch]) -> list[tuple[int, str]]:
    segments = []
    cursor = 0
    for match in sorted(covered_matches, key=lambda item: (item.start, item.end)):
        if cursor < match.start:
            segment = text[cursor : match.start]
            if segment.strip():
                segments.append((cursor, segment))
        cursor = max(cursor, match.end)
    if cursor < len(text):
        segment = text[cursor:]
        if segment.strip():
            segments.append((cursor, segment))
    return segments


def offset_match(match: PIIMatch, offset: int, text: str) -> PIIMatch:
    start = match.start + offset
    end = match.end + offset
    return PIIMatch(
        label=match.label,
        start=start,
        end=end,
        text=text[start:end],
        score=match.score,
        recognizer=match.recognizer,
    )


def recognize_redacted_chunks(
    recognizers: list[Recognizer],
    text: str,
    covered_matches: list[PIIMatch],
    *,
    max_workers: int | None = None,
) -> list[PIIMatch]:
    matches: list[PIIMatch] = []
    resolved_matches = resolve_overlaps(covered_matches)
    for offset, chunk in text_chunks(text):
        redacted_chunk, redacted_length = redact_chunk(chunk, offset, resolved_matches)
        if not has_ner_signal(redacted_chunk, redacted_length=redacted_length):
            continue
        chunk_matches = recognize_recognizer_group(
            recognizers,
            redacted_chunk,
            max_workers=max_workers,
        )
        for match in chunk_matches:
            shifted = offset_match(match, offset, text)
            if shifted.text.strip() and not any(
                spans_overlap(shifted, covered_match) for covered_match in resolved_matches
            ):
                matches.append(shifted)
    return matches


def text_chunks(text: str) -> list[tuple[int, str]]:
    chunks = []
    start = 0
    for index, character in enumerate(text):
        if character in ".!?\n\r;":
            end = index + 1
            chunk = text[start:end]
            if chunk.strip():
                chunks.append((start, chunk))
            start = end
    if start < len(text):
        chunk = text[start:]
        if chunk.strip():
            chunks.append((start, chunk))
    return chunks


def redact_chunk(chunk: str, offset: int, covered_matches: list[PIIMatch]) -> tuple[str, int]:
    characters = list(chunk)
    redacted_length = 0
    chunk_start = offset
    chunk_end = offset + len(chunk)
    for match in covered_matches:
        start = max(match.start, chunk_start) - offset
        end = min(match.end, chunk_end) - offset
        if start < end:
            characters[start:end] = " " * (end - start)
            redacted_length += end - start
    return "".join(characters), redacted_length


def has_ner_signal(
    text: str,
    *,
    redacted_length: int = 0,
    min_length: int = MIN_NER_CHUNK_LENGTH,
    max_redacted_ratio: float = MAX_REDACTED_CHUNK_RATIO,
) -> bool:
    if len(text.strip()) < min_length:
        return False
    if redacted_length / max(1, len(text)) > max_redacted_ratio:
        return False
    return any(character.isalpha() for character in text)


def recognize_recognizer_group(
    recognizers: list[Recognizer], text: str, *, max_workers: int | None = None
) -> list[PIIMatch]:
    if not recognizers:
        return []
    if len(recognizers) == 1 or max_workers == 1:
        return recognize_sequentially(recognizers, text)
    return recognize_concurrently(recognizers, text, max_workers=max_workers)


def is_ner_recognizer(recognizer: Recognizer) -> bool:
    return isinstance(recognizer, NERRecognizer)


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
