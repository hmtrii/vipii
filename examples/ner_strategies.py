"""Compare NER execution strategies without loading a real model.

Run from the repository after installing the package:

    python examples/ner_strategies.py
"""

from __future__ import annotations

from presidio_analyzer import EntityRecognizer, RecognizerResult

from vipii import Pattern, PatternRecognizer, PIIDetector

TEXT = "Số điện thoại 0912345678 của Nguyễn Văn A ở Hà Nội. CCCD 001203000123."
STRATEGIES = ["always", "fallback", "uncovered", "chunked", "never"]
ENTITIES = {"Nguyễn Văn A": "PERSON", "Hà Nội": "LOCATION"}


class FakeNerRecognizer(EntityRecognizer):
    """Stand in for a real model so this example needs no downloads.

    `vipii_is_ner = True` opts a recognizer into the detector's NER strategies.
    A real deployment would use `PIIDetector(ner_model=...)`, which builds
    Presidio's `HuggingFaceNerRecognizer`.
    """

    vipii_is_ner = True

    def __init__(self, seen_texts: list[str]) -> None:
        self.seen_texts = seen_texts
        super().__init__(
            supported_entities=sorted(set(ENTITIES.values())),
            name="ner",
            supported_language="vi",
        )

    def load(self) -> None:
        """No resources are required."""

    def analyze(self, text, entities, nlp_artifacts=None):  # type: ignore[no-untyped-def]
        del entities, nlp_artifacts
        self.seen_texts.append(text)
        results = []
        for phrase, label in ENTITIES.items():
            start = text.find(phrase)
            if start >= 0:
                results.append(
                    RecognizerResult(
                        entity_type=label,
                        start=start,
                        end=start + len(phrase),
                        score=0.98 if label == "PERSON" else 0.91,
                    )
                )
        return results


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
            FakeNerRecognizer(seen_texts),
        ],
        include_builtins=False,
        ner_strategy=strategy,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
