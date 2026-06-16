"""KFP step 2 — read raw CSV from S3, clean, write processed parquet to S3.

Mirrors src/component/data_transformer.py but is S3-in / S3-out so it composes
with the other KFP steps. Output is year-partitioned parquet at
s3://<bucket>/<processed-prefix>, the same location the Spark job and Katib use.
"""
import argparse
import logging

import pandas as pd

logger = logging.getLogger("worldbank.steps.transform")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["country.value", "date", "value"]].rename(columns={
        "country.value": "country",
        "date": "year",
        "value": "indicator_value",
    })
    df = df.dropna(subset=["indicator_value"])
    df["year"] = df["year"].astype(int)
    df["indicator_value"] = pd.to_numeric(df["indicator_value"], errors="coerce")
    return df.dropna(subset=["indicator_value"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--raw-prefix", default="worldbank/raw")
    p.add_argument("--processed-prefix", default="worldbank/processed")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    raw_uri = f"s3://{args.bucket}/{args.raw_prefix}/raw.csv"
    processed_uri = f"s3://{args.bucket}/{args.processed_prefix}"

    df = pd.read_csv(raw_uri)
    cleaned = clean(df)
    # partition_cols=year so re-runs upsert per-year partitions.
    cleaned.to_parquet(processed_uri, index=False, partition_cols=["year"])
    logger.info("Wrote %d cleaned rows to %s", len(cleaned), processed_uri)


if __name__ == "__main__":
    main()
