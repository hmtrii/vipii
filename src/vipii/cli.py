"""Command line interface for vipii."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vipii import PIIDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vipii", description="Vietnamese PII detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan UTF-8 text from a file path or literal text")
    scan.add_argument("input", help="File path or plain text to scan")
    scan.add_argument(
        "--input-type",
        choices=["auto", "file", "text"],
        default="auto",
        help="How to interpret the input argument. Defaults to auto.",
    )
    scan.add_argument("--format", choices=["text", "json"], default="text")
    scan.add_argument("--config", type=Path, help="YAML file with additional recognizers")
    scan.add_argument("--ner-model", help="Hugging Face token-classification model for NER")
    scan.add_argument("--ner-min-score", type=float, default=0.5, help="Minimum NER confidence")
    scan.add_argument(
        "--ner-strategy",
        choices=["always", "fallback", "uncovered", "chunked", "never"],
        default="always",
        help="Control when NER runs: always, fallback, uncovered, chunked, or never.",
    )
    scan.add_argument(
        "--redact", action="store_true", help="Print redacted text instead of match lines"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        return scan_input(
            args.input,
            input_type=args.input_type,
            output_format=args.format,
            redact=args.redact,
            config_path=args.config,
            ner_model=args.ner_model,
            ner_min_score=args.ner_min_score,
            ner_strategy=args.ner_strategy,
        )

    parser.error(f"unsupported command: {args.command}")
    return 2


def scan_input(
    value: str,
    *,
    input_type: str = "auto",
    output_format: str = "text",
    redact: bool = False,
    config_path: str | Path | None = None,
    ner_model: str | None = None,
    ner_min_score: float = 0.5,
    ner_strategy: str = "always",
) -> int:
    text = read_input(value, input_type=input_type)
    return scan_text(
        text,
        output_format=output_format,
        redact=redact,
        config_path=config_path,
        ner_model=ner_model,
        ner_min_score=ner_min_score,
        ner_strategy=ner_strategy,
    )


def scan_file(
    path: Path,
    *,
    output_format: str = "text",
    redact: bool = False,
    ner_strategy: str = "always",
) -> int:
    return scan_text(
        path.read_text(encoding="utf-8"),
        output_format=output_format,
        redact=redact,
        ner_strategy=ner_strategy,
    )


def scan_text(
    text: str,
    *,
    output_format: str = "text",
    redact: bool = False,
    config_path: str | Path | None = None,
    ner_model: str | None = None,
    ner_min_score: float = 0.5,
    ner_strategy: str = "always",
) -> int:
    detector = PIIDetector(
        config_path=config_path,
        ner_model=ner_model,
        ner_min_score=ner_min_score,
        ner_strategy=ner_strategy,
    )

    if redact:
        print(detector.redact(text))
        return 0

    matches = detector.detect(text)
    if output_format == "json":
        print(json.dumps([match.as_dict() for match in matches], ensure_ascii=False, indent=2))
        return 0

    for match in matches:
        print(
            f"{match.label}\t{match.start}:{match.end}\t{match.score:.2f}\t"
            f"{match.text}\t{match.recognizer or ''}"
        )
    return 0


def read_input(value: str, *, input_type: str = "auto") -> str:
    path = Path(value)
    if input_type == "file":
        return path.read_text(encoding="utf-8")
    if input_type == "text":
        return value
    if input_type == "auto" and path.is_file():
        return path.read_text(encoding="utf-8")
    if input_type == "auto":
        return value
    raise ValueError(f"unsupported input type: {input_type}")


if __name__ == "__main__":
    raise SystemExit(main())
