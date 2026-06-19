"""Benchmark local detection against the optional PySpark adapter.

Run from the repository after installing the package with Spark support:

    pip install -e ".[spark]"
    python examples/spark_benchmark.py --rows 50000 --partitions 4
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from time import perf_counter

from vipii import PIIDetector
from vipii.spark import with_pii_matches, with_redacted_column

SAMPLES = [
    "Khách hàng Nguyễn Văn A có số điện thoại 0912 345 678 và CCCD 001203000123.",
    "Liên hệ email an.nguyen@example.com, mã số thuế 0312345678, biển số 51F-123.45.",
    "Tài khoản ngân hàng 9704151234567890 cần được xác minh trước 16:30.",
    "Xin chào, nội dung này không chứa dữ liệu định danh rõ ràng.",
]


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    seconds: float
    rows: int
    matches: int | None = None

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.seconds if self.seconds else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000, help="Number of text rows to benchmark")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs for each engine",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=2,
        help="Number of Spark DataFrame partitions",
    )
    parser.add_argument(
        "--master",
        default="local[*]",
        help="Spark master URL, for example local[4]",
    )
    parser.add_argument(
        "--operation",
        choices=("detect", "redact"),
        default="detect",
        help="Operation to benchmark",
    )
    return parser.parse_args()


def make_texts(rows: int) -> list[str]:
    return [SAMPLES[index % len(SAMPLES)] for index in range(rows)]


def time_call(name: str, rows: int, callback: Callable[[], int]) -> BenchmarkResult:
    started = perf_counter()
    matches = callback()
    return BenchmarkResult(name=name, seconds=perf_counter() - started, rows=rows, matches=matches)


def benchmark_normal_detect(detector: PIIDetector, texts: Sequence[str]) -> int:
    return sum(len(detector.detect(text)) for text in texts)


def benchmark_normal_redact(detector: PIIDetector, texts: Sequence[str]) -> int:
    return sum(len(detector.redact(text)) for text in texts)


def build_spark_session(master: str):  # type: ignore[no-untyped-def]
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise SystemExit('Install Spark support first: pip install -e ".[spark]"') from exc

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    try:
        return (
            SparkSession.builder.master(master)
            .appName("vipii-spark-benchmark")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
    except Exception as exc:
        raise SystemExit(f"Spark session could not start: {exc}") from exc


def benchmark_spark_detect(dataframe) -> int:  # type: ignore[no-untyped-def]
    result = with_pii_matches(dataframe, input_col="text").selectExpr(
        "sum(size(pii_matches)) as total_matches"
    )
    return int(result.collect()[0].total_matches or 0)


def benchmark_spark_redact(dataframe) -> int:  # type: ignore[no-untyped-def]
    result = with_redacted_column(dataframe, input_col="text").selectExpr(
        "sum(length(redacted)) as total_length"
    )
    return int(result.collect()[0].total_length or 0)


def print_result(result: BenchmarkResult) -> None:
    suffix = "" if result.matches is None else f", output={result.matches}"
    print(
        f"{result.name:<14} {result.seconds:>8.3f}s  "
        f"{result.rows_per_second:>10,.0f} rows/s{suffix}"
    )


def print_fastest(title: str, results: Sequence[BenchmarkResult]) -> None:
    fastest = min(results, key=lambda result: result.seconds)
    print(f"{title:<14} fastest {fastest.seconds:.3f}s ({fastest.rows_per_second:,.0f} rows/s)")


def main() -> None:
    args = parse_args()
    if args.rows < 1:
        raise SystemExit("--rows must be at least 1")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.partitions < 1:
        raise SystemExit("--partitions must be at least 1")

    texts = make_texts(args.rows)
    detector = PIIDetector()
    spark = build_spark_session(args.master)

    try:
        dataframe = spark.createDataFrame([(text,) for text in texts], ["text"]).repartition(
            args.partitions
        )
        dataframe.cache()
        dataframe.count()

        if args.operation == "detect":
            normal_callback: Callable[[], int] = partial(benchmark_normal_detect, detector, texts)
            spark_callback: Callable[[], int] = partial(benchmark_spark_detect, dataframe)
        else:
            normal_callback = partial(benchmark_normal_redact, detector, texts)
            spark_callback = partial(benchmark_spark_redact, dataframe)

        print(
            f"Benchmarking {args.operation} on {args.rows:,} rows "
            f"({args.partitions} Spark partitions, {args.repeats} repeats)"
        )
        print()

        normal_callback()
        spark_callback()

        normal_results = []
        spark_results = []
        for run in range(1, args.repeats + 1):
            normal_result = time_call(f"normal #{run}", args.rows, normal_callback)
            spark_result = time_call(f"spark #{run}", args.rows, spark_callback)
            normal_results.append(normal_result)
            spark_results.append(spark_result)
            print_result(normal_result)
            print_result(spark_result)

        print()
        print_fastest("normal", normal_results)
        print_fastest("spark", spark_results)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
