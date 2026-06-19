"""Compare NER execution strategies without loading a real model.

Run from the repository after installing the package:

    python examples/ner_strategies.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vipii import NERRecognizer, Pattern, PatternRecognizer, PIIDetector

TEXT = "Số điện thoại 0912345678 của Nguyễn Văn A ở Hà Nội. CCCD 001203000123."
STRATEGIES = ["always", "fallback", "uncovered", "chunked", "never"]


def main() -> None:
    print("\nNER strategies")
    print("--------------")
    print(TEXT)

    for strategy in STRATEGIES:
        seen_texts: list[str] = []
        detector = detector_for_strategy(strategy, seen_texts)
        matches = detector.detect(TEXT)

        print(f"\nstrategy={strategy!r}")
        print("NER input:")
        if seen_texts:
            for seen_text in seen_texts:
                print(f"  {seen_text!r}")
        else:
            print("  skipped")

        print("Matches:")
        for match in matches:
            print(
                f"  {match.label:<14} {match.start:>2}:{match.end:<2} "
                f"score={match.score:.2f} text={match.text!r}"
            )


def detector_for_strategy(strategy: str, seen_texts: list[str]) -> PIIDetector:
    return PIIDetector(
        recognizers=[
            PatternRecognizer(
                name="phone_number",
                label="PHONE_NUMBER",
                patterns=[Pattern(label="PHONE_NUMBER", regex=r"\b0\d{9}\b")],
            ),
            PatternRecognizer(
                name="cccd",
                label="CCCD",
                patterns=[Pattern(label="CCCD", regex=r"\b\d{12}\b")],
            ),
            NERRecognizer(
                model_name="fake-vietnamese-ner",
                pipeline_factory=fake_pipeline_factory(seen_texts),
            ),
        ],
        include_builtins=False,
        ner_strategy=strategy,  # type: ignore[arg-type]
    )


def fake_pipeline_factory(
    seen_texts: list[str],
) -> Callable[..., Callable[[str], list[dict[str, Any]]]]:
    def factory(*args: Any, **kwargs: Any) -> Callable[[str], list[dict[str, Any]]]:
        def pipeline(text: str) -> list[dict[str, Any]]:
            seen_texts.append(text)
            return fake_entities(text)

        return pipeline

    return factory


def fake_entities(text: str) -> list[dict[str, Any]]:
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


if __name__ == "__main__":
    main()
