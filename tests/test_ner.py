from __future__ import annotations

import json

import pytest

from vipii import NERRecognizer, PIIDetector
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


def test_ner_requires_optional_dependency(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def missing_transformers(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError("Install vipii[ner] to use NERRecognizer")

    monkeypatch.setattr("vipii.recognizers.ner.load_transformers_pipeline", missing_transformers)
    recognizer = NERRecognizer(model_name="fake-vietnamese-ner")

    with pytest.raises(ImportError, match=r"vipii\[ner\]"):
        recognizer.recognize("Nguyễn Văn A")
