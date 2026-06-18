"""Optional PySpark adapter for DataFrame PII detection and redaction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vipii import PIIDetector

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame
    from pyspark.sql.udf import UserDefinedFunction

_DEFAULT_DETECTOR: PIIDetector | None = None


def _require_pyspark() -> tuple[Any, ...]:
    try:
        from pyspark.sql.functions import col, udf
        from pyspark.sql.types import (
            ArrayType,
            DoubleType,
            IntegerType,
            StringType,
            StructField,
            StructType,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without optional extra
        raise ImportError("Install vipii[spark] to use the PySpark adapter") from exc
    return col, udf, ArrayType, DoubleType, IntegerType, StringType, StructField, StructType


def _match_schema() -> Any:
    (
        _,
        _,
        ArrayType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    ) = _require_pyspark()
    return ArrayType(
        StructType(
            [
                StructField("label", StringType(), nullable=False),
                StructField("start", IntegerType(), nullable=False),
                StructField("end", IntegerType(), nullable=False),
                StructField("text", StringType(), nullable=False),
                StructField("score", DoubleType(), nullable=False),
                StructField("recognizer", StringType(), nullable=True),
            ]
        )
    )


def _default_detector() -> PIIDetector:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = PIIDetector()
    return _DEFAULT_DETECTOR


def detect_udf(detector: PIIDetector | None = None) -> UserDefinedFunction:
    """Create a UDF that returns detected PII matches as an array of structs."""

    _, udf, *_ = _require_pyspark()

    def detect_text(text: str | None) -> list[dict[str, object]]:
        if text is None:
            return []
        active_detector = detector or _default_detector()
        return [match.as_dict() for match in active_detector.detect(text)]

    return udf(detect_text, _match_schema())


def redact_udf(detector: PIIDetector | None = None, *, mask: str = "*") -> UserDefinedFunction:
    """Create a UDF that redacts detected PII in a string column."""

    _, udf, _, _, _, StringType, *_ = _require_pyspark()

    def redact_text(text: str | None) -> str | None:
        if text is None:
            return None
        active_detector = detector or _default_detector()
        return active_detector.redact(text, mask=mask)

    return udf(redact_text, StringType())


def with_pii_matches(
    dataframe: DataFrame,
    *,
    input_col: str | Column,
    output_col: str = "pii_matches",
    detector: PIIDetector | None = None,
) -> DataFrame:
    """Return a DataFrame with a column containing detected PII matches."""

    col, *_ = _require_pyspark()
    source = col(input_col) if isinstance(input_col, str) else input_col
    return dataframe.withColumn(output_col, detect_udf(detector)(source))


def with_redacted_column(
    dataframe: DataFrame,
    *,
    input_col: str | Column,
    output_col: str = "redacted",
    detector: PIIDetector | None = None,
    mask: str = "*",
) -> DataFrame:
    """Return a DataFrame with a column containing redacted text."""

    col, *_ = _require_pyspark()
    source = col(input_col) if isinstance(input_col, str) else input_col
    return dataframe.withColumn(output_col, redact_udf(detector, mask=mask)(source))


__all__ = [
    "detect_udf",
    "redact_udf",
    "with_pii_matches",
    "with_redacted_column",
]
