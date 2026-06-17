"""Run recognizers concurrently.

Run from the repository after installing the package:

    python examples/concurrent_detection.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter, sleep

from _helpers import print_matches

from vipii import PIIDetector
from vipii.models import PIIMatch


@dataclass
class SlowRegexRecognizer:
    name: str
    label: str
    regex: str
    delay_seconds: float = 0.5

    def recognize(self, text: str) -> list[PIIMatch]:
        sleep(self.delay_seconds)
        return [
            PIIMatch(
                label=self.label,
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                score=0.8,
                recognizer=self.name,
            )
            for match in re.finditer(self.regex, text)
        ]


def timed_detect(title: str, detector: PIIDetector, text: str) -> None:
    started = perf_counter()
    matches = detector.detect(text)
    elapsed = perf_counter() - started

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Detected {len(matches)} matches in {elapsed:.2f}s")
    for match in matches:
        print(
            f"{match.label:<14} {match.start:>2}:{match.end:<2} "
            f"score={match.score:.2f} text={match.text!r}"
        )


def main() -> None:
    text = "Mã khách hàng KH-123456 có mã đơn hàng DH-98765."
    recognizers = [
        SlowRegexRecognizer("customer_id", "CUSTOMER_ID", r"\bKH-\d{6}\b"),
        SlowRegexRecognizer("order_id", "ORDER_ID", r"\bDH-\d{5}\b"),
    ]

    concurrent_detector = PIIDetector(
        recognizers=recognizers,
        include_builtins=False,
    )
    sequential_detector = PIIDetector(
        recognizers=recognizers,
        include_builtins=False,
        max_workers=1,
    )

    print_matches("Concurrent recognizers", text, concurrent_detector)
    timed_detect("Concurrent timing", concurrent_detector, text)
    timed_detect("Sequential timing", sequential_detector, text)


if __name__ == "__main__":
    main()
