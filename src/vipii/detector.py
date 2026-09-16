"""Detector orchestration and redaction helpers."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal

from presidio_analyzer import (
    AnalyzerEngine,
    EntityRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)

from vipii.models import Pattern, PIIMatch
from vipii.nlp import VipiiNlpEngine, create_nlp_engine
from vipii.presidio import (
    LANGUAGE,
    build_analyzer,
    build_batch_analyzer,
    match_from_result,
    result_from_match,
)
from vipii.recognizers import (
    Recognizer,
    built_in_recognizers,
    custom_pattern_recognizer,
    ner_recognizer,
)

NERStrategy = Literal["always", "fallback", "uncovered", "chunked", "never"]
NER_STRATEGIES = {"always", "fallback", "uncovered", "chunked", "never"}
MIN_NER_CHUNK_LENGTH = 8
MAX_REDACTED_CHUNK_RATIO = 0.6


class PIIDetector:
    """Detect Vietnamese PII through a Presidio analyzer engine."""

    def __init__(
        self,
        recognizers: list[Recognizer] | None = None,
        *,
        config_path: str | Path | None = None,
        include_builtins: bool = True,
        ner_model: str | None = None,
        ner_min_score: float = 0.5,
        ner_strategy: NERStrategy = "always",
        score_threshold: float = 0.0,
        nlp_engine: str | VipiiNlpEngine | None = None,
    ) -> None:
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
                ner_recognizer(ner_model, min_score=ner_min_score),
            ]
        self.recognizers = list(recognizers)
        self.ner_strategy = ner_strategy
        self.score_threshold = score_threshold
        # Build once and share: tokenizers can be expensive to construct.
        self.nlp_engine = create_nlp_engine(nlp_engine)
        self._analyzer: AnalyzerEngine | None = None
        self._pattern_analyzer: AnalyzerEngine | None = None
        self._ner_analyzer: AnalyzerEngine | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, *, include_builtins: bool = True) -> PIIDetector:
        return cls(config_path=path, include_builtins=include_builtins)

    # -- recognizer views ------------------------------------------------

    @property
    def pattern_recognizers(self) -> list[Recognizer]:
        return [r for r in self.recognizers if not is_ner_recognizer(r)]

    @property
    def ner_recognizers(self) -> list[Recognizer]:
        return [r for r in self.recognizers if is_ner_recognizer(r)]

    # -- analyzers -------------------------------------------------------

    @property
    def analyzer(self) -> AnalyzerEngine:
        """Engine over every recognizer."""
        if self._analyzer is None:
            self._analyzer = self.build_engine(self.recognizers)
        return self._analyzer

    @property
    def pattern_analyzer(self) -> AnalyzerEngine:
        """Engine over the pattern recognizers alone.

        The NER strategies run patterns first, then decide whether — and on what
        text — to run the model. Presidio cannot restrict ``analyze()`` to part of
        a registry, so that subset needs its own engine. When there is no NER
        recognizer to leave out, the full engine is reused.
        """
        if not self.ner_recognizers:
            return self.analyzer
        if self._pattern_analyzer is None:
            self._pattern_analyzer = self.build_engine(self.pattern_recognizers)
        return self._pattern_analyzer

    @property
    def ner_analyzer(self) -> AnalyzerEngine:
        """Engine over the model-backed recognizers alone."""
        if self._ner_analyzer is None:
            self._ner_analyzer = self.build_engine(self.ner_recognizers)
        return self._ner_analyzer

    def build_engine(self, recognizers: list[Recognizer]) -> AnalyzerEngine:
        return build_analyzer(
            recognizers,
            nlp_engine=self.nlp_engine,
            default_score_threshold=self.score_threshold,
        )

    @property
    def registry(self) -> RecognizerRegistry:
        return self.analyzer.registry

    def supported_entities(self) -> list[str]:
        return self.analyzer.get_supported_entities(language=LANGUAGE)

    def _invalidate(self) -> None:
        self._analyzer = None
        self._pattern_analyzer = None
        self._ner_analyzer = None

    # -- configuration ---------------------------------------------------

    def add_pattern(self, pattern: Pattern) -> None:
        self.add_recognizer(custom_pattern_recognizer(pattern))

    def add_recognizer(self, recognizer: Recognizer) -> None:
        self.recognizers.append(recognizer)
        self._invalidate()

    def add_ner_model(self, model_name: str, *, min_score: float = 0.5, **kwargs: object) -> None:
        self.add_recognizer(ner_recognizer(model_name, min_score=min_score, **kwargs))

    # -- detection -------------------------------------------------------

    def analyze(
        self, text: str, *, analyzer: AnalyzerEngine | None = None, **kwargs: object
    ) -> list[RecognizerResult]:
        """Run Presidio and return raw results. Defaults to the full engine."""
        engine = self.analyzer if analyzer is None else analyzer
        return engine.analyze(text=text, language=LANGUAGE, **kwargs)  # type: ignore[arg-type]

    def detect(
        self,
        text: str,
        *,
        entities: list[str] | None = None,
        score_threshold: float | None = None,
        allow_list: list[str] | None = None,
        allow_list_match: str = "exact",
        ad_hoc_recognizers: list[EntityRecognizer] | None = None,
        context: list[str] | None = None,
        return_decision_process: bool = False,
    ) -> list[PIIMatch]:
        if not self.recognizers and not ad_hoc_recognizers:
            return []
        options = {
            "entities": entities,
            "score_threshold": score_threshold,
            "allow_list": allow_list,
            "allow_list_match": allow_list_match,
            "ad_hoc_recognizers": ad_hoc_recognizers,
            "context": context,
            "return_decision_process": return_decision_process,
        }
        if self.ner_strategy == "always":
            candidates = self.detect_matches(text, self.analyzer, options)
        else:
            candidates = self.detect_with_ner_strategy(text, options)
        return resolve_overlaps(candidates)

    def detect_batch(self, texts: Sequence[str], **kwargs: object) -> list[list[PIIMatch]]:
        """Detect PII across many texts.

        Uses Presidio's ``BatchAnalyzerEngine`` for the default ``always``
        strategy; the segment-based NER strategies are applied per text.
        """
        if self.ner_strategy != "always" or kwargs:
            return [self.detect(text, **kwargs) for text in texts]  # type: ignore[arg-type]
        batch = build_batch_analyzer(self.analyzer)
        results = batch.analyze_iterator(texts=list(texts), language=LANGUAGE)
        return [
            resolve_overlaps([match_from_result(text, result) for result in text_results])
            for text, text_results in zip(texts, results, strict=True)
        ]

    def detect_with_ner_strategy(self, text: str, options: dict[str, object]) -> list[PIIMatch]:
        strategy = self.ner_strategy
        candidates = self.detect_matches(text, self.pattern_analyzer, options)
        if strategy == "never" or not self.ner_recognizers:
            return candidates
        if strategy == "fallback":
            if candidates:
                return candidates
            return self.detect_ner(text, options)
        if strategy == "uncovered":
            return [*candidates, *self.recognize_uncovered_text(text, candidates, options)]
        if strategy == "chunked":
            return [*candidates, *self.recognize_redacted_chunks(text, candidates, options)]
        return candidates

    def detect_matches(
        self, text: str, analyzer: AnalyzerEngine, options: dict[str, object]
    ) -> list[PIIMatch]:
        return [
            match_from_result(text, result)
            for result in self.analyze(text, analyzer=analyzer, **options)
        ]

    def detect_ner(self, text: str, options: dict[str, object]) -> list[PIIMatch]:
        return self.detect_matches(text, self.ner_analyzer, options)

    def recognize_uncovered_text(
        self, text: str, covered_matches: list[PIIMatch], options: dict[str, object]
    ) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        for offset, segment in uncovered_text_segments(text, resolve_overlaps(covered_matches)):
            for match in self.detect_ner(segment, options):
                matches.append(offset_match(match, offset, text))
        return matches

    def recognize_redacted_chunks(
        self, text: str, covered_matches: list[PIIMatch], options: dict[str, object]
    ) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        resolved_matches = resolve_overlaps(covered_matches)
        for offset, chunk in text_chunks(text):
            redacted_chunk, redacted_length = redact_chunk(chunk, offset, resolved_matches)
            if not has_ner_signal(redacted_chunk, redacted_length=redacted_length):
                continue
            for match in self.detect_ner(redacted_chunk, options):
                shifted = offset_match(match, offset, text)
                if shifted.text.strip() and not any(
                    spans_overlap(shifted, covered_match) for covered_match in resolved_matches
                ):
                    matches.append(shifted)
        return matches

    # -- redaction -------------------------------------------------------

    def redact(self, text: str, mask: str = "*", *, operators: dict | None = None) -> str:
        """Redact detected spans with Presidio's anonymizer.

        Defaults to masking every character of each span, matching vipii's
        historical behaviour. Pass ``operators`` to use any Presidio operator
        (``replace``, ``hash``, ``encrypt``, ...).
        """
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        matches = self.detect(text)
        if not matches:
            return text
        if operators is None:
            operators = {
                "DEFAULT": OperatorConfig(
                    "mask",
                    {"masking_char": mask, "chars_to_mask": len(text), "from_end": False},
                )
            }
        result = AnonymizerEngine().anonymize(
            text=text,
            analyzer_results=[result_from_match(match) for match in matches],
            operators=operators,
        )
        return result.text


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


@lru_cache(maxsize=1)
def ner_recognizer_types() -> tuple[type, ...]:
    """Presidio recognizer classes the NER strategies treat as model-backed.

    Presidio imports these lazily with ``None`` sentinels when transformers/torch
    are absent, so missing optional dependencies are not an error here.
    """
    import presidio_analyzer.predefined_recognizers as predefined

    types: list[type] = []

    for name in ("HuggingFaceNerRecognizer", "SpacyRecognizer", "GLiNERRecognizer"):
        candidate = getattr(predefined, name, None)
        if isinstance(candidate, type):
            types.append(candidate)
    return tuple(types)


def is_ner_recognizer(recognizer: Recognizer) -> bool:
    """Report whether a recognizer is model-backed rather than pattern-backed.

    Set ``vipii_is_ner`` on a recognizer to override the classification.
    """
    override = getattr(recognizer, "vipii_is_ner", None)
    if override is not None:
        return bool(override)
    return isinstance(recognizer, ner_recognizer_types())


def resolve_overlaps(matches: list[PIIMatch]) -> list[PIIMatch]:
    """Resolve cross-entity span overlaps.

    Presidio's ``remove_duplicates`` only collapses containment within a single
    entity type, so vipii still arbitrates between different labels covering the
    same span.
    """
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
