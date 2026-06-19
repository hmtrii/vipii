from __future__ import annotations

import os

import pytest

from vipii.spark import with_pii_matches, with_redacted_column

pyspark_sql = pytest.importorskip("pyspark.sql")


@pytest.fixture(scope="module")
def spark():  # type: ignore[no-untyped-def]
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    try:
        session = (
            pyspark_sql.SparkSession.builder.master("local[1]")
            .appName("vipii-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
    except Exception as exc:
        pytest.skip(f"Spark session could not start: {exc}")
    try:
        yield session
    finally:
        session.stop()


def test_with_pii_matches_detects_dataframe_text_column(spark) -> None:  # type: ignore[no-untyped-def]
    dataframe = spark.createDataFrame(
        [("Số điện thoại 0912345678.",), ("Xin chào.",), (None,)],
        ["text"],
    )

    rows = with_pii_matches(dataframe, input_col="text").collect()

    assert rows[0].pii_matches[0].label == "PHONE_NUMBER"
    assert rows[0].pii_matches[0].text == "0912345678"
    assert rows[1].pii_matches == []
    assert rows[2].pii_matches == []


def test_with_redacted_column_redacts_dataframe_text_column(spark) -> None:  # type: ignore[no-untyped-def]
    dataframe = spark.createDataFrame(
        [("CCCD 001203000123.",), (None,)],
        ["text"],
    )

    rows = with_redacted_column(dataframe, input_col="text", mask="#").collect()

    assert rows[0].redacted == "CCCD ############."
    assert rows[1].redacted is None
