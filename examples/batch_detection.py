"""Scan many texts with Presidio's batch analyzer.

Run from the repository after installing the package:

    python examples/batch_detection.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from vipii import PIIDetector

TEXTS = [
    "Khách hàng A có số điện thoại 0912 345 678.",
    "Khách hàng B có CCCD 001203000123.",
    "Liên hệ email khach@example.vn để được hỗ trợ.",
    "Không có dữ liệu định danh trong câu này.",
    "Biển số xe 51F-123.45 và mã số thuế 0312345678.",
]


def print_results(title: str, texts: list[str], results: list[list]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for text, matches in zip(texts, results, strict=True):
        print(text)
        for match in matches:
            print(
                f"  {match.label:<14} {match.start:>2}:{match.end:<2} "
                f"score={match.score:.2f} text={match.text!r}"
            )


def main() -> None:
    detector = PIIDetector()

    started = perf_counter()
    batched = detector.detect_batch(TEXTS)
    batch_elapsed = perf_counter() - started

    print_results("Batch detection", TEXTS, batched)
    print(f"\ndetect_batch: {batch_elapsed:.3f}s for {len(TEXTS)} texts")

    # PIIDetector holds no per-call state, so detect() is safe to fan out across
    # threads once the detector is fully configured. Do not add recognizers while
    # scans are in flight.
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        threaded = list(executor.map(detector.detect, TEXTS))
    threaded_elapsed = perf_counter() - started

    print(f"thread pool : {threaded_elapsed:.3f}s for {len(TEXTS)} texts")
    assert threaded == batched


if __name__ == "__main__":
    main()
