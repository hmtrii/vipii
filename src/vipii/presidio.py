"""Presidio engine wiring: NLP engine, context enhancement, and result mapping."""

from __future__ import annotations

from collections.abc import Iterable

from presidio_analyzer import (
    AnalyzerEngine,
    BatchAnalyzerEngine,
    EntityRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.context_aware_enhancers import ContextAwareEnhancer
from presidio_analyzer.nlp_engine import NlpArtifacts

from vipii.models import PIIMatch
from vipii.nlp import LANGUAGE, VipiiNlpEngine, create_nlp_engine

__all__ = [
    "LANGUAGE",
    "NoopEntityRecognizer",
    "VipiiContextAwareEnhancer",
    "build_analyzer",
    "build_batch_analyzer",
    "build_registry",
    "create_nlp_engine",
    "match_from_result",
    "result_from_match",
]


class VipiiContextAwareEnhancer(ContextAwareEnhancer):
    """Disable Presidio's global enhancer.

    vipii boosts scores inside each recognizer instead, because context words and
    token windows are configured per pattern and per recognizer, which Presidio's
    engine-wide :class:`LemmaContextAwareEnhancer` cannot express.
    """

    def __init__(self) -> None:
        super().__init__(
            context_similarity_factor=0.0,
            min_score_with_context_similarity=0.0,
            context_prefix_count=0,
            context_suffix_count=0,
        )

    def enhance_using_context(
        self,
        text: str,
        raw_results: list[RecognizerResult],
        nlp_artifacts: NlpArtifacts,
        recognizers: list[EntityRecognizer],
        context: list[str] | None = None,
    ) -> list[RecognizerResult]:
        del text, nlp_artifacts, recognizers, context
        return raw_results


class NoopEntityRecognizer(EntityRecognizer):
    """Keep an intentionally empty registry from loading Presidio's defaults.

    ``AnalyzerEngine.__init__`` calls ``registry.load_predefined_recognizers()``
    whenever ``registry.recognizers`` is falsy, which would silently add US and EU
    recognizers such as ``CREDIT_CARD`` and ``US_SSN`` to a Vietnamese detector.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entities=["VIPII_NOOP"],
            supported_language=LANGUAGE,
            name="VipiiNoopRecognizer",
        )

    def load(self) -> None:
        """No resources are required."""

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        del text, entities, nlp_artifacts
        return []


def build_registry(recognizers: Iterable[EntityRecognizer]) -> RecognizerRegistry:
    return RecognizerRegistry(
        recognizers=list(recognizers) or [NoopEntityRecognizer()],
        supported_languages=[LANGUAGE],
    )


def build_analyzer(
    recognizers: Iterable[EntityRecognizer],
    *,
    nlp_engine: str | VipiiNlpEngine | None = None,
    default_score_threshold: float = 0.0,
) -> AnalyzerEngine:
    return AnalyzerEngine(
        registry=build_registry(recognizers),
        nlp_engine=create_nlp_engine(nlp_engine),
        supported_languages=[LANGUAGE],
        context_aware_enhancer=VipiiContextAwareEnhancer(),
        default_score_threshold=default_score_threshold,
    )


def build_batch_analyzer(analyzer: AnalyzerEngine) -> BatchAnalyzerEngine:
    return BatchAnalyzerEngine(analyzer_engine=analyzer)


def result_from_match(match: PIIMatch) -> RecognizerResult:
    return RecognizerResult(
        entity_type=match.label,
        start=match.start,
        end=match.end,
        score=match.score,
        recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: match.recognizer},
    )


def match_from_result(text: str, result: RecognizerResult) -> PIIMatch:
    metadata = result.recognition_metadata or {}
    return PIIMatch(
        label=result.entity_type,
        start=result.start,
        end=result.end,
        text=text[result.start : result.end],
        score=result.score,
        recognizer=metadata.get(RecognizerResult.RECOGNIZER_NAME_KEY),
    )
