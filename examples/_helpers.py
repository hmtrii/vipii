"""Shared helpers for example scripts."""

from __future__ import annotations

from vipii import PIIDetector


def print_matches(title: str, text: str, detector: PIIDetector) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(text)
    for match in detector.detect(text):
        print(
            f"{match.label:<14} {match.start:>2}:{match.end:<2} "
            f"score={match.score:.2f} text={match.text!r}"
        )
