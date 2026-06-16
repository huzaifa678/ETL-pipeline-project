"""PySpark ETL job for the World Bank dataset, run by the Spark Operator.

The driver fetches the raw data from the World Bank API (small, bounded payload),
hands it to Spark for cleaning/typing, and writes year-partitioned parquet to S3.
Mirrors the logic in src/component/{data_ingestion,data_transformer}.py.

Dates are NOT hardcoded:
  * --end-year defaults to the current year (use 0 / omit for "now").
  * --incremental reads the last ingested year from a state object in S3 and only
    fetches newer years, so a cron schedule keeps the dataset live.
Writing is partitioned by year with dynamic partition overwrite, so re-running a
year upserts that partition instead of duplicating or wiping history.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from urllib.parse import urlparse

import boto3
import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType, StringType

logger = logging.getLogger("worldbank.spark.etl")


def _s3_client():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))


def read_last_year(state_uri: str, default: int) -> int:
    """Return the last ingested year from the S3 state object, or `default`."""
    if not state_uri:
        return default
    parsed = urlparse(state_uri)
    try:
        obj = _s3_client().get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        return int(obj["Body"].read().decode().strip())
    except Exception:
        return default


def write_last_year(state_uri: str, year: int) -> None:
    if not state_uri:
        return
    parsed = urlparse(state_uri)
    _s3_client().put_object(
        Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), Body=str(year).encode()
    )


def fetch_records(base_url: str, countries: str, indicator: str,
                  start_year: int, end_year: int) -> list[dict]:
    """Page through the World Bank API and return flat records."""
    url = f"{base_url}/{countries}/indicator/{indicator}"
    params = {"format": "json", "per_page": 1000, "date": f"{start_year}:{end_year}"}
    records, page = [], 1
    while True:
        params["page"] = page
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data or len(data) < 2 or data[1] is None:
            break
        for row in data[1]:
            records.append({
                "country": (row.get("country") or {}).get("value"),
                "year": row.get("date"),
                "indicator_value": row.get("value"),
            })
        if page >= data[0].get("pages", 1):
            break
        page += 1
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.getenv("BASE_URL",
                   "http://api.worldbank.org/v2/countries"))
    p.add_argument("--indicator", default="SP.POP.TOTL")
    p.add_argument("--countries", default="br;cn;us;de")
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=0,
                   help="0 or omitted = current year")
    p.add_argument("--incremental", action="store_true",
                   help="only fetch years after the last ingested year (from S3 state)")
    p.add_argument("--state-uri", default="",
                   help="s3://bucket/key tracking the last ingested year")
    p.add_argument("--output", required=True,
                   help="s3a://bucket/prefix for processed parquet")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    end_year = args.end_year if args.end_year > 0 else datetime.now().year
    start_year = args.start_year
    if args.incremental:
        last = read_last_year(args.state_uri, default=args.start_year - 1)
        start_year = last + 1

    if start_year > end_year:
        logger.info("No new data: start_year %d > end_year %d. Skipping.",
                    start_year, end_year)
        return

    spark = (
        SparkSession.builder.appName("worldbank-etl")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        # Upsert only the year partitions present in this run.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )

    schema = StructType([
        StructField("country", StringType(), True),
        StructField("year", StringType(), True),
        StructField("indicator_value", StringType(), True),
    ])

    records = fetch_records(args.base_url, args.countries, args.indicator,
                            start_year, end_year)
    if not records:
        logger.warning("World Bank API returned no rows for %d:%d. Skipping.",
                       start_year, end_year)
        spark.stop()
        return

    df = spark.createDataFrame(records, schema=schema)
    cleaned = (
        df.dropna(subset=["indicator_value"])
        .withColumn("year", F.col("year").cast(IntegerType()))
        .withColumn("indicator_value", F.col("indicator_value").cast("double"))
        .dropna(subset=["indicator_value", "year"])
    )

    cleaned.write.mode("overwrite").partitionBy("year").parquet(args.output)
    logger.info("Wrote %d rows for %d:%d to %s",
                cleaned.count(), start_year, end_year, args.output)

    write_last_year(args.state_uri, end_year)
    spark.stop()


if __name__ == "__main__":
    main()
