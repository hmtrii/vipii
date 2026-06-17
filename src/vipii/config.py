"""YAML recognizer configuration loading."""

from __future__ import annotations

import ast
from importlib import resources
from pathlib import Path
from typing import Any

from vipii.models import Pattern
from vipii.recognizers import PatternRecognizer, validator_by_name

BUILTIN_CONFIG = "builtin_recognizers.yml"


def load_builtin_recognizers() -> list[PatternRecognizer]:
    with resources.files("vipii").joinpath(BUILTIN_CONFIG).open(encoding="utf-8") as config:
        return recognizers_from_config(load_yaml_text(config.read(), source=BUILTIN_CONFIG))


def load_recognizers_from_yaml(path: str | Path) -> list[PatternRecognizer]:
    path = Path(path)
    return recognizers_from_config(
        load_yaml_text(path.read_text(encoding="utf-8"), source=str(path))
    )


def recognizers_from_config(config: dict[str, Any]) -> list[PatternRecognizer]:
    recognizer_items = config.get("recognizers")
    if not isinstance(recognizer_items, list):
        raise ValueError("recognizer config must contain a 'recognizers' list")

    recognizers = []
    for item in recognizer_items:
        if not isinstance(item, dict):
            raise ValueError("each recognizer entry must be a mapping")
        recognizers.append(recognizer_from_config(item))
    return recognizers


def recognizer_from_config(config: dict[str, Any]) -> PatternRecognizer:
    name = required_string(config, "name")
    label = required_string(config, "label")
    validator_name = optional_string(config, "validator")
    token_window = int(config.get("token_window", 8))
    pattern_items = config.get("patterns")
    if not isinstance(pattern_items, list) or not pattern_items:
        raise ValueError(f"recognizer '{name}' must define at least one pattern")

    validator = validator_by_name(validator_name) if validator_name else None
    patterns = [
        Pattern(
            label=str(pattern.get("label", label)),
            regex=required_string(pattern, "regex"),
            context_words=string_list(pattern.get("context_words", [])),
            base_score=float(pattern.get("base_score", 0.5)),
            recognizer=name,
        )
        for pattern in pattern_items
        if isinstance(pattern, dict)
    ]
    if len(patterns) != len(pattern_items):
        raise ValueError(f"recognizer '{name}' contains a non-mapping pattern entry")
    return PatternRecognizer(
        name=name,
        label=label,
        patterns=patterns,
        validator=validator,
        token_window=token_window,
    )


def load_yaml_text(text: str, *, source: str = "<string>") -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        parsed = parse_simple_yaml(text, source=source)
    else:
        parsed = yaml.safe_load(text) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{source} must contain a top-level mapping")
    return parsed


def required_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string")
    return value


def optional_string(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' must be a non-empty string when set")
    return value


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("context_words must be a list of strings")
    return value


def parse_simple_yaml(text: str, *, source: str = "<string>") -> Any:
    lines = []
    for number, raw in enumerate(text.splitlines(), start=1):
        content = strip_yaml_comment(raw).rstrip()
        if not content.strip():
            continue
        lines.append((number, len(content) - len(content.lstrip(" ")), content.lstrip(" ")))
    if not lines:
        return {}
    value, index = parse_block(lines, 0, lines[0][1], source)
    if index != len(lines):
        line_number = lines[index][0]
        raise ValueError(f"{source}:{line_number}: unexpected trailing YAML content")
    return value


def parse_block(
    lines: list[tuple[int, int, str]], index: int, indent: int, source: str
) -> tuple[Any, int]:
    if lines[index][2].startswith("- "):
        return parse_list(lines, index, indent, source)
    return parse_mapping(lines, index, indent, source)


def parse_mapping(
    lines: list[tuple[int, int, str]], index: int, indent: int, source: str
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(lines):
        line_number, line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"{source}:{line_number}: unexpected indentation")
        if content.startswith("- "):
            break
        key, value = parse_key_value(content, source, line_number)
        index += 1
        if value is None:
            if index >= len(lines) or lines[index][1] <= indent:
                mapping[key] = {}
            else:
                mapping[key], index = parse_block(lines, index, lines[index][1], source)
        else:
            mapping[key] = parse_scalar(value)
    return mapping, index


def parse_list(
    lines: list[tuple[int, int, str]], index: int, indent: int, source: str
) -> tuple[list[Any], int]:
    items = []
    while index < len(lines):
        line_number, line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent or not content.startswith("- "):
            break
        tail = content[2:].strip()
        index += 1
        if not tail:
            if index >= len(lines) or lines[index][1] <= indent:
                items.append(None)
            else:
                item, index = parse_block(lines, index, lines[index][1], source)
                items.append(item)
            continue
        if ":" in tail and not tail.startswith(("'", '"')):
            key, value = parse_key_value(tail, source, line_number)
            item = {key: parse_scalar(value) if value is not None else {}}
            if index < len(lines) and lines[index][1] > indent:
                nested, index = parse_mapping(lines, index, lines[index][1], source)
                item.update(nested)
            items.append(item)
            continue
        items.append(parse_scalar(tail))
    return items, index


def parse_key_value(content: str, source: str, line_number: int) -> tuple[str, str | None]:
    if ":" not in content:
        raise ValueError(f"{source}:{line_number}: expected 'key: value'")
    key, value = content.split(":", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"{source}:{line_number}: empty mapping key")
    value = value.strip()
    return key, value or None


def parse_scalar(value: str | None) -> Any:
    if value is None:
        return None
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if value.startswith("[") or value.startswith('"'):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def strip_yaml_comment(line: str) -> str:
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#":
            return line[:index]
    return line
