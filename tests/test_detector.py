from __future__ import annotations

from threading import Barrier

import pytest

from vipii import Pattern, PIIDetector
from vipii.models import PIIMatch


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


class BarrierRecognizer:
    def __init__(self, name: str, barrier: Barrier) -> None:
        self.name = name
        self.barrier = barrier

    def recognize(self, text: str) -> list[PIIMatch]:
        self.barrier.wait(timeout=5)
        return [
            PIIMatch(
                label=self.name.upper(),
                start=0,
                end=len(text),
                text=text,
                score=0.5,
                recognizer=self.name,
            )
        ]


def test_detect_runs_recognizers_concurrently() -> None:
    barrier = Barrier(2)
    detector = PIIDetector(
        recognizers=[
            BarrierRecognizer("first", barrier),
            BarrierRecognizer("second", barrier),
        ],
        include_builtins=False,
    )

    matches = detector.detect("abc")

    assert [match.label for match in matches] == ["SECOND"]


def test_detector_rejects_invalid_max_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        PIIDetector(max_workers=0)
