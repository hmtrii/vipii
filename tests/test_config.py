from __future__ import annotations

from vipii import PIIDetector, load_recognizers_from_yaml
from vipii.config import load_yaml_text


def test_loads_custom_recognizer_from_yaml(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "recognizers.yml"
    config.write_text(
        """
recognizers:
  - name: customer_id
    label: CUSTOMER_ID
    patterns:
      - regex: '\\bKH-\\d{6}\\b'
        context_words: ["mã khách hàng"]
        base_score: 0.6
""",
        encoding="utf-8",
    )

    detector = PIIDetector.from_yaml(config, include_builtins=False)
    matches = detector.detect("Mã khách hàng KH-123456.")

    assert len(matches) == 1
    assert matches[0].label == "CUSTOMER_ID"
    assert matches[0].recognizer == "customer_id"
    assert matches[0].score > 0.6


def test_config_appends_to_builtin_recognizers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "recognizers.yml"
    config.write_text(
        """
recognizers:
  - name: ticket_id
    label: TICKET_ID
    patterns:
      - regex: '\\bTK-\\d{4}\\b'
        context_words: ["mã phiếu"]
        base_score: 0.6
""",
        encoding="utf-8",
    )

    labels = {
        match.label
        for match in PIIDetector(config_path=config).detect(
            "Mã phiếu TK-1234 có số điện thoại 0912345678."
        )
    }

    assert labels == {"TICKET_ID", "PHONE_NUMBER"}


def test_load_recognizers_public_helper(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "recognizers.yml"
    config.write_text(
        """
recognizers:
  - name: order_code
    label: ORDER_CODE
    patterns:
      - regex: '\\bOD\\d{5}\\b'
""",
        encoding="utf-8",
    )

    recognizers = load_recognizers_from_yaml(config)

    assert recognizers[0].name == "order_code"
    assert recognizers[0].label == "ORDER_CODE"


def test_fallback_yaml_parser_preserves_regex_backslashes() -> None:
    config = load_yaml_text(
        """
recognizers:
  - name: sample
    label: SAMPLE
    patterns:
      - regex: '\\bA\\d{2}\\b'
""",
    )

    assert config["recognizers"][0]["patterns"][0]["regex"] == r"\bA\d{2}\b"
