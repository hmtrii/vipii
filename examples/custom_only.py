"""Run a detector with only custom recognizers.

Run from the repository after installing the package:

    python examples/custom_only.py
"""

from __future__ import annotations

from _helpers import print_matches

from vipii import Pattern, PIIDetector


def main() -> None:
    detector = PIIDetector(include_builtins=False)
    detector.add_pattern(Pattern(label="ORDER_ID", regex=r"\bDH-\d{5}\b"))
    text = "Đơn hàng DH-12345 của số điện thoại 0912345678."

    print_matches("Custom-only detector", text, detector)


if __name__ == "__main__":
    main()
