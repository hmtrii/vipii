"""Add a custom regex pattern in Python.

Run from the repository after installing the package:

    python examples/custom_pattern.py
"""

from __future__ import annotations

from _helpers import print_matches

from vipii import Pattern, PIIDetector


def main() -> None:
    detector = PIIDetector()
    detector.add_pattern(
        Pattern(
            label="CUSTOMER_ID",
            regex=r"\bKH-\d{6}\b",
            context_words=["mã khách hàng", "customer id"],
            base_score=0.6,
        )
    )
    text = "Mã khách hàng KH-123456 có số điện thoại 0912 345 678."

    print_matches("Custom pattern from Python", text, detector)


if __name__ == "__main__":
    main()
