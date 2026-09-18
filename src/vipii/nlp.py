"""Vietnamese NLP engines backing Presidio's analyzer.

Presidio requires an ``NlpEngine`` even when every recognizer is regex-based. The
engine supplies the tokens that context scoring measures its window against, so
the choice of tokenizer changes how far "nearby" reaches around a match.

``underthesea`` is the default because it segments Vietnamese compounds into
single tokens: ``"điện thoại"`` and ``"khách hàng"`` are one token each rather
than two, which keeps a token window centred on meaningful words.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine

from vipii.scoring import TOKEN_RE

LANGUAGE = "vi"
DEFAULT_NLP_ENGINE = "underthesea"


@dataclass(frozen=True)
class Token:
    """Minimal spaCy-compatible token."""

    text: str
    idx: int

    @property
    def lemma_(self) -> str:
        """Surface form; Vietnamese is not inflected and has no lemmatizer here."""
        return self.text

    @property
    def end(self) -> int:
        return self.idx + len(self.text)


class VipiiNlpEngine(NlpEngine):
    """Base engine that turns a tokenizer into Presidio ``NlpArtifacts``."""

    name = "vipii"

    def tokenize(self, text: str) -> list[Token]:
        raise NotImplementedError

    def load(self) -> None:
        """No external model is required by default."""

    def is_loaded(self) -> bool:
        return True

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        tokens = self.tokenize(text)
        return NlpArtifacts(
            entities=[],
            tokens=tokens,  # type: ignore[arg-type]
            tokens_indices=[token.idx for token in tokens],
            lemmas=[token.text for token in tokens],
            nlp_engine=self,
            language=language,
        )

    def process_batch(
        self,
        texts: Iterable[str],
        language: str,
        batch_size: int = 1,
        n_process: int = 1,
        **kwargs: Any,
    ) -> Iterator[tuple[str, NlpArtifacts]]:
        del batch_size, n_process, kwargs
        for text in texts:
            yield text, self.process_text(text, language)

    def is_stopword(self, word: str, language: str) -> bool:
        del word, language
        return False

    def is_punct(self, word: str, language: str) -> bool:
        del language
        return not any(character.isalnum() for character in word)

    def get_supported_entities(self) -> list[str]:
        return []

    def get_supported_languages(self) -> list[str]:
        return [LANGUAGE]


class RegexNlpEngine(VipiiNlpEngine):
    """Dependency-free engine splitting on Unicode word characters."""

    name = "regex"

    def tokenize(self, text: str) -> list[Token]:
        return [Token(text=match.group(0), idx=match.start()) for match in TOKEN_RE.finditer(text)]


class UnderthesaNlpEngine(VipiiNlpEngine):
    """Engine backed by ``underthesea.word_tokenize``.

    underthesea returns words without offsets, so tokens are realigned against the
    source text by a forward scan. Token normalization is disabled to keep surface
    forms alignable; any token that cannot be located is dropped rather than
    shifting every later offset.
    """

    name = "underthesea"

    def __init__(self, tokenizer: Callable[..., list[str]] | None = None) -> None:
        self._tokenizer = tokenizer

    def load(self) -> None:
        self.tokenizer()

    def tokenizer(self) -> Callable[..., list[str]]:
        if self._tokenizer is None:
            try:
                from underthesea import word_tokenize
            except ImportError as exc:  # pragma: no cover - exercised without the dependency
                raise ImportError(
                    "underthesea is required for the default NLP engine. "
                    "Install it, or use PIIDetector(nlp_engine='regex')."
                ) from exc
            self._tokenizer = word_tokenize
        return self._tokenizer

    def tokenize(self, text: str) -> list[Token]:
        words = self.tokenizer()(text, use_token_normalize=False)
        return align_tokens(text, words)


def align_tokens(text: str, words: Iterable[str]) -> list[Token]:
    """Locate each word in the source text, preserving order."""
    tokens = []
    cursor = 0
    for word in words:
        index = text.find(word, cursor)
        if index < 0:
            continue
        tokens.append(Token(text=word, idx=index))
        cursor = index + len(word)
    return tokens


NLP_ENGINES: dict[str, type[VipiiNlpEngine]] = {
    "underthesea": UnderthesaNlpEngine,
    "regex": RegexNlpEngine,
}


def create_nlp_engine(engine: str | VipiiNlpEngine | None = None) -> VipiiNlpEngine:
    """Build an NLP engine by name.

    Accepts a name from :data:`NLP_ENGINES`, an already-built engine (returned
    unchanged), or ``None`` for :data:`DEFAULT_NLP_ENGINE`.
    """
    if engine is None:
        engine = DEFAULT_NLP_ENGINE
    if isinstance(engine, NlpEngine):
        return engine
    if engine not in NLP_ENGINES:
        raise ValueError(
            f"unknown nlp_engine {engine!r}; expected one of {sorted(NLP_ENGINES)} "
            "or an NlpEngine instance"
        )
    return NLP_ENGINES[engine]()


def token_spans(nlp_artifacts: NlpArtifacts | None) -> list[tuple[int, int]] | None:
    """Return ``(start, end)`` spans for engine tokens, or ``None`` if unavailable."""
    if nlp_artifacts is None or not nlp_artifacts.tokens:
        return None
    return [(token.idx, token.idx + len(token.text)) for token in nlp_artifacts.tokens]
