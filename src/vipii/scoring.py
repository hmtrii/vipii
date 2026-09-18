"""Lightweight Vietnamese context scoring."""

from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    return value.casefold()


def regex_token_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in TOKEN_RE.finditer(text)]


def context_window(
    text: str,
    start: int,
    end: int,
    token_window: int = 8,
    *,
    spans: list[tuple[int, int]] | None = None,
) -> str:
    """Return the text spanning ``token_window`` tokens either side of a match.

    ``spans`` comes from the active NLP engine when available, so Vietnamese
    compounds tokenized as one word count as one token. Falls back to a Unicode
    word regex.
    """
    tokens = regex_token_spans(text) if spans is None else spans
    if not tokens:
        return ""

    before = [idx for idx, (_, token_end) in enumerate(tokens) if token_end <= start]
    after = [idx for idx, (token_start, _) in enumerate(tokens) if token_start >= end]

    left_idx = max((before[-1] + 1 if before else 0) - token_window, 0)
    right_idx = min((after[0] if after else len(tokens)) + token_window, len(tokens))

    return text[tokens[left_idx][0] : tokens[right_idx - 1][1]]


def score_with_context(
    text: str,
    start: int,
    end: int,
    base_score: float,
    context_words: tuple[str, ...],
    *,
    token_window: int = 8,
    boost: float = 0.2,
    spans: list[tuple[int, int]] | None = None,
) -> float:
    if not context_words:
        return base_score

    window = normalize_text(
        context_window(text, start, end, token_window=token_window, spans=spans)
    )
    hits = sum(1 for word in context_words if normalize_text(word) in window)
    return min(1.0, base_score + hits * boost)
