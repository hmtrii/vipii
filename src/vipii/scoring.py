"""Lightweight Vietnamese context scoring."""

from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    return value.casefold()


def context_window(text: str, start: int, end: int, token_window: int = 8) -> str:
    tokens = list(TOKEN_RE.finditer(text))
    before = [idx for idx, token in enumerate(tokens) if token.end() <= start]
    after = [idx for idx, token in enumerate(tokens) if token.start() >= end]

    left_idx = max((before[-1] + 1 if before else 0) - token_window, 0)
    right_idx = min((after[0] if after else len(tokens)) + token_window, len(tokens))
    if not tokens:
        return ""

    return text[tokens[left_idx].start() : tokens[right_idx - 1].end()]


def score_with_context(
    text: str,
    start: int,
    end: int,
    base_score: float,
    context_words: tuple[str, ...],
    *,
    token_window: int = 8,
    boost: float = 0.2,
) -> float:
    if not context_words:
        return base_score

    window = normalize_text(context_window(text, start, end, token_window=token_window))
    hits = sum(1 for word in context_words if normalize_text(word) in window)
    return min(1.0, base_score + hits * boost)
