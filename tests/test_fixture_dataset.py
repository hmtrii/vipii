from __future__ import annotations

import json
from pathlib import Path

from vipii import PIIDetector

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_vietnamese_pii.jsonl"


def test_fixture_dataset_has_mvp_size() -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]

    assert len(rows) >= 50


def test_fixture_dataset_expected_labels_are_detected() -> None:
    detector = PIIDetector()
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]

    for row in rows:
        detected = {match.label for match in detector.detect(row["text"])}
        assert set(row["labels"]).issubset(detected), row["text"]
