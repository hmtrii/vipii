"""Vietnamese regex recognizer built on Presidio's PatternRecognizer."""

from __future__ import annotations

import regex
from presidio_analyzer import Pattern as PresidioPattern
from presidio_analyzer import PatternRecognizer as PresidioPatternRecognizer
from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from vipii.models import Pattern
from vipii.nlp import LANGUAGE, token_spans
from vipii.recognizers.validators import Validator
from vipii.scoring import score_with_context

# Presidio defaults to regex.I|M|S. vipii patterns are authored against
# re.IGNORECASE only; DOTALL and MULTILINE would change what `.`, `^` and `$`
# match, so pin the flags to preserve pattern semantics.
REGEX_FLAGS = regex.IGNORECASE


class PatternRecognizer(PresidioPatternRecognizer):
    """Presidio recognizer with vipii validators and per-pattern context scoring."""

    def __init__(
        self,
        name: str,
        label: str,
        patterns: list[Pattern],
        validator: Validator | None = None,
        token_window: int = 8,
    ) -> None:
        if not patterns:
            raise ValueError(f"recognizer '{name}' must define at least one pattern")
        labels = {pattern.label for pattern in patterns}
        if labels != {label}:
            raise ValueError(
                f"recognizer '{name}' must emit a single entity; "
                f"expected {label!r} but patterns declare {sorted(labels)!r}"
            )

        self.label = label
        self.validator = validator
        self.token_window = token_window
        self.vipii_patterns = {
            pattern_name(name, index): pattern for index, pattern in enumerate(patterns)
        }

        super().__init__(
            supported_entity=label,
            name=name,
            supported_language=LANGUAGE,
            patterns=[
                PresidioPattern(
                    name=pattern_name(name, index),
                    regex=pattern.regex,
                    score=pattern.base_score,
                )
                for index, pattern in enumerate(patterns)
            ],
            context=sorted(
                {word for pattern in patterns for word in (pattern.context_words or ())}
            ),
            global_regex_flags=REGEX_FLAGS,
        )

    def invalidate_result(self, pattern_text: str) -> bool | None:
        """Drop matches the validator rejects.

        Uses invalidation rather than ``validate_result`` because a successful
        ``validate_result`` forces the score to ``EntityRecognizer.MAX_SCORE``,
        which would discard the pattern's base score and context boosting.
        """
        if self.validator is None:
            return None
        return not self.validator(pattern_text)

    def enhance_using_context(
        self,
        text: str,
        raw_recognizer_results: list[RecognizerResult],
        other_raw_recognizer_results: list[RecognizerResult],
        nlp_artifacts: NlpArtifacts,
        context: list[str] | None = None,
    ) -> list[RecognizerResult]:
        del other_raw_recognizer_results
        spans = token_spans(nlp_artifacts)
        for result in raw_recognizer_results:
            pattern = self.vipii_patterns.get(matched_pattern_name(result))
            context_words = tuple(pattern.context_words or ()) if pattern else ()
            if context:
                context_words = (*context_words, *context)
            if not context_words:
                continue
            score = score_with_context(
                text,
                result.start,
                result.end,
                result.score,
                context_words,
                token_window=self.token_window,
                spans=spans,
            )
            if score > result.score:
                result.score = score
                result.recognition_metadata[RecognizerResult.IS_SCORE_ENHANCED_BY_CONTEXT_KEY] = (
                    True
                )
        return raw_recognizer_results


def pattern_name(recognizer_name: str, index: int) -> str:
    return f"{recognizer_name}-{index}"


def matched_pattern_name(result: RecognizerResult) -> str | None:
    explanation = result.analysis_explanation
    return explanation.pattern_name if explanation else None


def custom_pattern_recognizer(pattern: Pattern) -> PatternRecognizer:
    name = pattern.recognizer or "custom"
    return PatternRecognizer(name=name, label=pattern.label, patterns=[pattern])
