from __future__ import annotations

import pytest

from vipii import PIIDetector, create_nlp_engine
from vipii.nlp import (
    DEFAULT_NLP_ENGINE,
    NLP_ENGINES,
    RegexNlpEngine,
    UnderthesaNlpEngine,
    align_tokens,
)

SENTENCE = "Số điện thoại khách hàng là 0912 345 678."


def test_underthesea_is_the_default_engine() -> None:
    assert DEFAULT_NLP_ENGINE == "underthesea"
    assert isinstance(create_nlp_engine(), UnderthesaNlpEngine)
    assert isinstance(PIIDetector().nlp_engine, UnderthesaNlpEngine)


def test_factory_builds_engines_by_name() -> None:
    assert set(NLP_ENGINES) == {"underthesea", "regex"}
    assert isinstance(create_nlp_engine("regex"), RegexNlpEngine)
    assert isinstance(create_nlp_engine("underthesea"), UnderthesaNlpEngine)


def test_factory_passes_through_engine_instances() -> None:
    engine = RegexNlpEngine()

    assert create_nlp_engine(engine) is engine


def test_factory_rejects_unknown_engines() -> None:
    with pytest.raises(ValueError, match="unknown nlp_engine"):
        create_nlp_engine("spacy")


@pytest.mark.parametrize("name", sorted(NLP_ENGINES))
def test_tokens_align_with_source_text(name: str) -> None:
    tokens = create_nlp_engine(name).tokenize(SENTENCE)

    assert tokens
    assert all(SENTENCE[token.idx : token.end] == token.text for token in tokens)
    assert [token.idx for token in tokens] == sorted(token.idx for token in tokens)


def test_underthesea_segments_vietnamese_compounds() -> None:
    texts = {token.text for token in UnderthesaNlpEngine().tokenize(SENTENCE)}

    assert "điện thoại" in texts
    assert "khách hàng" in texts


def test_regex_engine_splits_compounds_into_syllables() -> None:
    texts = {token.text for token in RegexNlpEngine().tokenize(SENTENCE)}

    assert {"điện", "thoại"} <= texts


def test_align_tokens_drops_words_missing_from_text() -> None:
    tokens = align_tokens("a b c", ["a", "zz", "c"])

    assert [(token.text, token.idx) for token in tokens] == [("a", 0), ("c", 4)]


def test_detector_accepts_engine_name_and_instance() -> None:
    by_name = PIIDetector(nlp_engine="regex")
    by_instance = PIIDetector(nlp_engine=RegexNlpEngine())

    assert by_name.detect(SENTENCE) == by_instance.detect(SENTENCE)


def test_word_segmentation_widens_the_context_window() -> None:
    # MST uses token_window=3. "Mã số thuế" only fits inside that window when
    # Vietnamese compounds are segmented as single tokens.
    text = "Mã số thuế chi nhánh 0312345678-001 nằm trong hợp đồng."

    segmented = PIIDetector(nlp_engine="underthesea").detect(text)[0]
    syllables = PIIDetector(nlp_engine="regex").detect(text)[0]

    assert segmented.label == syllables.label == "MST"
    assert segmented.score > syllables.score
