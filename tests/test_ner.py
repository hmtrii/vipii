from __future__ import annotations

import json

import pytest

from vipii import NERRecognizer, Pattern, PatternRecognizer, PIIDetector
from vipii.cli import main


def fake_pipeline_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
    def pipeline(text: str) -> list[dict[str, object]]:
        return [
            {
                "entity_group": "PER",
                "start": text.index("Nguyễn Văn A"),
                "end": text.index("Nguyễn Văn A") + len("Nguyễn Văn A"),
                "score": 0.98,
            },
            {
                "entity_group": "LOC",
                "start": text.index("Hà Nội"),
                "end": text.index("Hà Nội") + len("Hà Nội"),
                "score": 0.91,
            },
        ]

    return pipeline


def test_ner_recognizer_converts_model_entities_to_pii_matches() -> None:
    recognizer = NERRecognizer(
        model_name="fake-vietnamese-ner",
        pipeline_factory=fake_pipeline_factory,
    )

    matches = recognizer.recognize("Nguyễn Văn A sống tại Hà Nội.")

    assert [match.label for match in matches] == ["PERSON", "LOCATION"]
    assert [match.text for match in matches] == ["Nguyễn Văn A", "Hà Nội"]
    assert all(match.recognizer == "ner" for match in matches)


def test_ner_recognizer_filters_low_confidence_entities() -> None:
    def low_score_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        return lambda text: [
            {
                "entity_group": "PER",
                "start": 0,
                "end": len("Nguyễn"),
                "score": 0.2,
            }
        ]

    recognizer = NERRecognizer(
        model_name="fake-vietnamese-ner",
        min_score=0.8,
        pipeline_factory=low_score_factory,
    )

    assert recognizer.recognize("Nguyễn Văn A") == []


def test_detector_accepts_ner_recognizer() -> None:
    detector = PIIDetector(
        recognizers=[
            NERRecognizer(
                model_name="fake-vietnamese-ner",
                pipeline_factory=fake_pipeline_factory,
            )
        ],
        include_builtins=False,
    )

    matches = detector.detect("Nguyễn Văn A sống tại Hà Nội.")

    assert [match.label for match in matches] == ["PERSON", "LOCATION"]


def test_detector_skips_ner_when_fallback_patterns_match() -> None:
    def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("NER should not run when fallback patterns match")

    detector = PIIDetector(
        recognizers=[
            PatternRecognizer(
                name="customer_id",
                label="CUSTOMER_ID",
                patterns=[Pattern(label="CUSTOMER_ID", regex=r"\bKH-\d{6}\b")],
            ),
            NERRecognizer(
                model_name="fake-vietnamese-ner",
                pipeline_factory=fail_if_called,
            ),
        ],
        include_builtins=False,
        ner_strategy="fallback",
    )

    matches = detector.detect("Mã khách hàng KH-123456.")

    assert [match.label for match in matches] == ["CUSTOMER_ID"]


def test_detector_runs_ner_when_fallback_patterns_do_not_match() -> None:
    detector = PIIDetector(
        recognizers=[
            PatternRecognizer(
                name="customer_id",
                label="CUSTOMER_ID",
                patterns=[Pattern(label="CUSTOMER_ID", regex=r"\bKH-\d{6}\b")],
            ),
            NERRecognizer(
                model_name="fake-vietnamese-ner",
                pipeline_factory=fake_pipeline_factory,
            ),
        ],
        include_builtins=False,
        ner_strategy="fallback",
    )

    matches = detector.detect("Nguyễn Văn A sống tại Hà Nội.")

    assert [match.label for match in matches] == ["PERSON", "LOCATION"]


def test_detector_never_strategy_skips_ner() -> None:
    def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("NER should not run with never strategy")

    detector = PIIDetector(
        recognizers=[
            PatternRecognizer(
                name="customer_id",
                label="CUSTOMER_ID",
                patterns=[Pattern(label="CUSTOMER_ID", regex=r"\bKH-\d{6}\b")],
            ),
            NERRecognizer(
                model_name="fake-vietnamese-ner",
                pipeline_factory=fail_if_called,
            ),
        ],
        include_builtins=False,
        ner_strategy="never",
    )

    matches = detector.detect("Mã khách hàng KH-123456 và Nguyễn Văn A.")

    assert [match.label for match in matches] == ["CUSTOMER_ID"]


def test_detector_uncovered_strategy_runs_ner_outside_pattern_spans() -> None:
    seen_texts = []

    def tolerant_pipeline_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        def pipeline(text: str) -> list[dict[str, object]]:
            seen_texts.append(text)
            entities = []
            if "Nguyễn Văn A" in text:
                entities.append(
                    {
                        "entity_group": "PER",
                        "start": text.index("Nguyễn Văn A"),
                        "end": text.index("Nguyễn Văn A") + len("Nguyễn Văn A"),
                        "score": 0.98,
                    }
                )
            if "Hà Nội" in text:
                entities.append(
                    {
                        "entity_group": "LOC",
                        "start": text.index("Hà Nội"),
                        "end": text.index("Hà Nội") + len("Hà Nội"),
                        "score": 0.91,
                    }
                )
            return entities

        return pipeline

    text = "Số điện thoại 0912345678 của Nguyễn Văn A ở Hà Nội."
    detector = PIIDetector(
        recognizers=[
            PatternRecognizer(
                name="phone_number",
                label="PHONE_NUMBER",
                patterns=[Pattern(label="PHONE_NUMBER", regex=r"\b0\d{9}\b")],
            ),
            NERRecognizer(
                model_name="fake-vietnamese-ner",
                pipeline_factory=tolerant_pipeline_factory,
            ),
        ],
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
    assert all("0912345678" not in seen_text for seen_text in seen_texts)


def test_detector_rejects_invalid_ner_strategy() -> None:
    with pytest.raises(ValueError, match="ner_strategy"):
        PIIDetector(ner_strategy="sometimes")  # type: ignore[arg-type]


def test_cli_uses_ner_model(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("vipii.recognizers.ner.load_transformers_pipeline", fake_pipeline_factory)

    assert (
        main(
            [
                "scan",
                "Nguyễn Văn A sống tại Hà Nội.",
                "--ner-model",
                "fake-vietnamese-ner",
                "--format",
                "json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert {item["label"] for item in payload} == {"PERSON", "LOCATION"}


def test_cli_can_skip_ner_with_fallback_strategy(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("NER should not run when fallback patterns match")

    monkeypatch.setattr("vipii.recognizers.ner.load_transformers_pipeline", fail_if_called)

    assert (
        main(
            [
                "scan",
                "Số điện thoại 0912345678.",
                "--ner-model",
                "fake-vietnamese-ner",
                "--ner-strategy",
                "fallback",
                "--format",
                "json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert [item["label"] for item in payload] == ["PHONE_NUMBER"]


def test_cli_accepts_never_ner_strategy(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("NER should not run with never strategy")

    monkeypatch.setattr("vipii.recognizers.ner.load_transformers_pipeline", fail_if_called)

    assert (
        main(
            [
                "scan",
                "Số điện thoại 0912345678.",
                "--ner-model",
                "fake-vietnamese-ner",
                "--ner-strategy",
                "never",
                "--format",
                "json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert [item["label"] for item in payload] == ["PHONE_NUMBER"]


def test_ner_requires_optional_dependency(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def missing_transformers(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError("Install vipii[ner] to use NERRecognizer")

    monkeypatch.setattr("vipii.recognizers.ner.load_transformers_pipeline", missing_transformers)
    recognizer = NERRecognizer(model_name="fake-vietnamese-ner")

    with pytest.raises(ImportError, match=r"vipii\[ner\]"):
        recognizer.recognize("Nguyễn Văn A")
