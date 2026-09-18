from __future__ import annotations

from presidio_analyzer import (
    AnalyzerEngine,
    EntityRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)

from vipii import Pattern, PatternRecognizer, PIIDetector


def labels(text: str) -> set[str]:
    return {match.label for match in PIIDetector().detect(text)}


def test_public_api_and_custom_pattern() -> None:
    detector = PIIDetector()
    detector.add_pattern(
        Pattern(label="CUSTOMER_ID", regex=r"\bKH-\d{6}\b", context_words=["mã khách hàng"])
    )

    matches = detector.detect("Mã khách hàng KH-123456 có số điện thoại 0912 345 678.")

    assert [match.label for match in matches] == ["CUSTOMER_ID", "PHONE_NUMBER"]
    assert matches[0].score > 0.5


def test_builtin_recognizers_cover_structured_vietnamese_pii() -> None:
    text = (
        "CCCD 001203000123, CMND 123456789, số điện thoại 0912 345 678, "
        "email khach@example.vn, ngày sinh 02/09/1990, mã số thuế 0312345678, "
        "số BHXH 1234567890, thẻ BHYT DN1234567890123, hộ chiếu B1234567, "
        "biển số xe 51F-123.45, GPLX 012345678901, địa chỉ IP 192.168.1.10, "
        "mã thiết bị 550e8400-e29b-41d4-a716-446655440000, "
        "thẻ 9704 0000 1234 5678 và tài khoản ngân hàng 123456789012."
    )

    assert labels(text) == {
        "CCCD",
        "CMND",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "DATE_OF_BIRTH",
        "MST",
        "SOCIAL_INSURANCE_NUMBER",
        "HEALTH_INSURANCE_NUMBER",
        "PASSPORT",
        "VEHICLE_PLATE",
        "DRIVER_LICENSE",
        "IP_ADDRESS",
        "DEVICE_ID",
        "BANK_CARD",
        "BANK_ACCOUNT",
    }


def test_context_boosting_raises_confidence() -> None:
    bare = PIIDetector().detect("Liên hệ 0912345678.")[0]
    contextual = PIIDetector().detect("Số điện thoại khách hàng là 0912345678.")[0]

    assert contextual.score > bare.score


def test_resolves_overlaps_by_score() -> None:
    matches = PIIDetector().detect("CCCD 001203000123.")

    assert len(matches) == 1
    assert matches[0].label == "CCCD"


def test_context_specific_identifiers_win_digit_overlaps() -> None:
    matches = PIIDetector().detect("Số BHXH 1234567890 và GPLX 001203000123.")

    assert [match.label for match in matches] == ["SOCIAL_INSURANCE_NUMBER", "DRIVER_LICENSE"]


def test_redaction_masks_detected_spans() -> None:
    text = "Số điện thoại 0912345678."

    assert PIIDetector().redact(text) == "Số điện thoại **********."


def test_vehicle_plate_requires_official_prefix() -> None:
    detector = PIIDetector()

    valid = detector.detect("Biển số xe 51F-123.45 và biển kiểm soát 80A-12345.")
    invalid = detector.detect("Biển số xe 13A-12345 không hợp lệ.")

    assert [match.text for match in valid] == ["51F-123.45", "80A-12345"]
    assert invalid == []


def test_no_matches_returns_empty_list() -> None:
    assert PIIDetector().detect("Không có dữ liệu định danh trong câu này.") == []


def test_detector_can_be_created_without_builtin_recognizers() -> None:
    assert PIIDetector(include_builtins=False).detect("Số điện thoại 0912345678.") == []


class RecordingRecognizer(EntityRecognizer):
    def __init__(self, name: str, calls: list[str]) -> None:
        self.calls = calls
        super().__init__(supported_entities=[name.upper()], name=name, supported_language="vi")

    def load(self) -> None:
        """No resources are required."""

    def analyze(self, text, entities, nlp_artifacts=None):  # type: ignore[no-untyped-def]
        self.calls.append(self.name)
        return [
            RecognizerResult(
                entity_type=self.name.upper(),
                start=0,
                end=len(text),
                score=0.5,
            )
        ]


def test_detect_runs_recognizers_through_presidio() -> None:
    calls: list[str] = []
    detector = PIIDetector(
        recognizers=[
            RecordingRecognizer("first", calls),
            RecordingRecognizer("second", calls),
        ],
        include_builtins=False,
    )

    matches = detector.detect("abc")

    assert isinstance(detector.analyzer, AnalyzerEngine)
    assert isinstance(detector.registry, RecognizerRegistry)
    assert set(calls) == {"first", "second"}
    assert [match.label for match in matches] == ["SECOND"]


def test_builtin_recognizers_are_native_presidio_recognizers() -> None:
    from presidio_analyzer import PatternRecognizer as PresidioPatternRecognizer

    recognizers = PIIDetector().registry.recognizers

    assert recognizers
    assert all(isinstance(r, PresidioPatternRecognizer) for r in recognizers)


def test_empty_detector_does_not_inherit_presidio_default_recognizers() -> None:
    entities = set(PIIDetector(include_builtins=False).supported_entities())

    assert not entities & {"CREDIT_CARD", "US_SSN", "IBAN_CODE", "EMAIL_ADDRESS"}


def test_detect_filters_to_requested_entities() -> None:
    text = "Số điện thoại 0912345678 và email khach@example.vn."

    matches = PIIDetector().detect(text, entities=["EMAIL_ADDRESS"])

    assert [match.label for match in matches] == ["EMAIL_ADDRESS"]


def test_detect_honours_allow_list() -> None:
    text = "Liên hệ khach@example.vn."

    assert PIIDetector().detect(text, allow_list=["khach@example.vn"]) == []


def test_detect_honours_score_threshold() -> None:
    text = "Liên hệ 0912345678."

    assert PIIDetector().detect(text, score_threshold=0.99) == []


def test_detect_batch_matches_per_text_detection() -> None:
    texts = [
        "Số điện thoại 0912345678.",
        "CCCD 001203000123.",
        "Không có dữ liệu định danh.",
    ]
    detector = PIIDetector()

    assert detector.detect_batch(texts) == [detector.detect(text) for text in texts]


def test_redact_accepts_presidio_operators() -> None:
    from presidio_anonymizer.entities import OperatorConfig

    redacted = PIIDetector().redact(
        "Số điện thoại 0912345678.",
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"})},
    )

    assert redacted == "Số điện thoại <PII>."


def test_pattern_analyzer_is_reused_when_there_is_no_ner_recognizer() -> None:
    detector = PIIDetector()

    assert detector.ner_recognizers == []
    assert detector.pattern_analyzer is detector.analyzer


def test_recognizer_views_split_patterns_from_ner() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from _ner_helpers import StubNerRecognizer

    pattern = PatternRecognizer(
        name="phone_number",
        label="PHONE_NUMBER",
        patterns=[Pattern(label="PHONE_NUMBER", regex=r"\b0\d{9}\b")],
    )
    ner = StubNerRecognizer()
    detector = PIIDetector(recognizers=[pattern, ner], include_builtins=False)

    assert detector.pattern_recognizers == [pattern]
    assert detector.ner_recognizers == [ner]
    assert detector.pattern_analyzer is not detector.analyzer
    assert detector.ner_analyzer is not detector.analyzer


def test_adding_a_recognizer_rebuilds_the_analyzers() -> None:
    detector = PIIDetector()
    before = detector.analyzer

    detector.add_pattern(Pattern(label="CUSTOMER_ID", regex=r"\bKH-\d{6}\b"))

    assert detector.analyzer is not before
    assert [m.label for m in detector.detect("Mã KH-123456.")] == ["CUSTOMER_ID"]


def test_analyze_returns_raw_presidio_results() -> None:
    from presidio_analyzer import RecognizerResult as PresidioResult

    results = PIIDetector().analyze("Số điện thoại 0912345678.")

    assert results
    assert all(isinstance(result, PresidioResult) for result in results)
