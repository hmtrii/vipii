from __future__ import annotations

import pytest
from _ner_helpers import (
    ExplodingNerRecognizer,
    StubNerRecognizer,
    presidio_hf_recognizer,
)

from vipii import Pattern, PatternRecognizer, PIIDetector, ner_recognizer
from vipii.detector import has_ner_signal, is_ner_recognizer
from vipii.recognizers.ner import VIETNAMESE_LABEL_MAPPING

TEXT = "Nguyễn Văn A sống tại Hà Nội."


def phone_recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="phone_number",
        label="PHONE_NUMBER",
        patterns=[Pattern(label="PHONE_NUMBER", regex=r"\b0\d{9}\b")],
    )


def customer_recognizer() -> PatternRecognizer:
    return PatternRecognizer(
        name="customer_id",
        label="CUSTOMER_ID",
        patterns=[Pattern(label="CUSTOMER_ID", regex=r"\bKH-\d{6}\b")],
    )


# -- the Presidio recognizer vipii configures --------------------------------


def test_ner_recognizer_builds_a_presidio_recognizer() -> None:
    from presidio_analyzer.predefined_recognizers import HuggingFaceNerRecognizer

    recognizer = presidio_hf_recognizer([])

    assert isinstance(recognizer, HuggingFaceNerRecognizer)
    assert recognizer.supported_language == "vi"
    assert recognizer.name == "ner"
    assert recognizer.threshold == 0.5


def test_ner_recognizer_maps_vietnamese_labels() -> None:
    recognizer = presidio_hf_recognizer([])

    assert recognizer.label_mapping == VIETNAMESE_LABEL_MAPPING
    assert "ADDRESS" in set(recognizer.label_mapping.values())


def test_ner_recognizer_validates_arguments() -> None:
    with pytest.raises(ValueError, match="model_name"):
        ner_recognizer("")
    with pytest.raises(ValueError, match="min_score"):
        ner_recognizer("fake-vietnamese-ner", min_score=1.5)


def test_ner_recognizer_converts_model_entities_to_pii_matches() -> None:
    recognizer = presidio_hf_recognizer(
        [
            {"entity_group": "PER", "start": 0, "end": 12, "score": 0.98},
            {"entity_group": "LOC", "start": 22, "end": 28, "score": 0.91},
        ]
    )
    detector = PIIDetector(recognizers=[recognizer], include_builtins=False)

    matches = detector.detect(TEXT)

    assert [match.label for match in matches] == ["PERSON", "LOCATION"]
    assert [match.text for match in matches] == ["Nguyễn Văn A", "Hà Nội"]


def test_ner_recognizer_filters_low_confidence_entities() -> None:
    recognizer = presidio_hf_recognizer(
        [{"entity_group": "PER", "start": 0, "end": 12, "score": 0.2}],
        min_score=0.8,
    )
    detector = PIIDetector(recognizers=[recognizer], include_builtins=False)

    assert detector.detect(TEXT) == []


# -- strategy classification -------------------------------------------------


def test_presidio_ner_recognizers_participate_in_strategies() -> None:
    assert is_ner_recognizer(presidio_hf_recognizer([]))


def test_pattern_recognizers_are_not_ner() -> None:
    assert not is_ner_recognizer(phone_recognizer())


def test_ner_classification_can_be_overridden() -> None:
    recognizer = phone_recognizer()
    recognizer.vipii_is_ner = True

    assert is_ner_recognizer(recognizer)


# -- strategies --------------------------------------------------------------


def test_has_ner_signal_uses_length_and_redacted_ratio() -> None:
    assert has_ner_signal("Nguyễn Văn A", redacted_length=0)
    assert not has_ner_signal("Hà", redacted_length=0)
    assert not has_ner_signal("CCCD             .", redacted_length=12)
    assert has_ner_signal("Số điện thoại            của Nguyễn Văn A.", redacted_length=10)


def test_detector_accepts_ner_recognizer() -> None:
    detector = PIIDetector(recognizers=[StubNerRecognizer()], include_builtins=False)

    assert [match.label for match in detector.detect(TEXT)] == ["PERSON", "LOCATION"]


def test_detector_skips_ner_when_fallback_patterns_match() -> None:
    detector = PIIDetector(
        recognizers=[customer_recognizer(), ExplodingNerRecognizer()],
        include_builtins=False,
        ner_strategy="fallback",
    )

    assert [m.label for m in detector.detect("Mã khách hàng KH-123456.")] == ["CUSTOMER_ID"]


def test_detector_runs_ner_when_fallback_patterns_do_not_match() -> None:
    detector = PIIDetector(
        recognizers=[customer_recognizer(), StubNerRecognizer()],
        include_builtins=False,
        ner_strategy="fallback",
    )

    assert [match.label for match in detector.detect(TEXT)] == ["PERSON", "LOCATION"]


def test_detector_never_strategy_skips_ner() -> None:
    detector = PIIDetector(
        recognizers=[customer_recognizer(), ExplodingNerRecognizer()],
        include_builtins=False,
        ner_strategy="never",
    )

    matches = detector.detect("Mã khách hàng KH-123456 và Nguyễn Văn A.")

    assert [match.label for match in matches] == ["CUSTOMER_ID"]


def test_presidio_ner_recognizer_is_skipped_by_never_strategy() -> None:
    def explode(text: str) -> list[dict[str, object]]:
        raise AssertionError("NER should not run with never strategy")

    recognizer = presidio_hf_recognizer([])
    recognizer.ner_pipeline = explode
    detector = PIIDetector(
        recognizers=[customer_recognizer(), recognizer],
        include_builtins=False,
        ner_strategy="never",
    )

    assert [m.label for m in detector.detect("Mã khách hàng KH-123456.")] == ["CUSTOMER_ID"]


def test_detector_uncovered_strategy_runs_ner_outside_pattern_spans() -> None:
    seen: list[str] = []
    text = "Số điện thoại 0912345678 của Nguyễn Văn A ở Hà Nội."
    detector = PIIDetector(
        recognizers=[phone_recognizer(), StubNerRecognizer(seen_texts=seen)],
        include_builtins=False,
        ner_strategy="uncovered",
    )

    matches = detector.detect(text)

    assert [match.label for match in matches] == ["PHONE_NUMBER", "PERSON", "LOCATION"]
    assert [match.text for match in matches] == ["0912345678", "Nguyễn Văn A", "Hà Nội"]
    assert [(match.start, match.end) for match in matches] == [
        (text.index("0912345678"), text.index("0912345678") + len("0912345678")),
        (text.index("Nguyễn Văn A"), text.index("Nguyễn Văn A") + len("Nguyễn Văn A")),
        (text.index("Hà Nội"), text.index("Hà Nội") + len("Hà Nội")),
    ]
    assert all("0912345678" not in seen_text for seen_text in seen)


def test_detector_chunked_strategy_redacts_patterns_before_ner() -> None:
    seen: list[str] = []
    text = "Số điện thoại 0912345678 của Nguyễn Văn A ở Hà Nội. CCCD 001203000123."
    detector = PIIDetector(
        recognizers=[
            phone_recognizer(),
            PatternRecognizer(
                name="cccd", label="CCCD", patterns=[Pattern(label="CCCD", regex=r"\b\d{12}\b")]
            ),
            StubNerRecognizer(seen_texts=seen),
        ],
        include_builtins=False,
        ner_strategy="chunked",
    )

    matches = detector.detect(text)

    assert [match.label for match in matches] == [
        "PHONE_NUMBER",
        "PERSON",
        "LOCATION",
        "CCCD",
    ]
    assert seen
    assert all("0912345678" not in seen_text for seen_text in seen)
    assert all("001203000123" not in seen_text for seen_text in seen)
