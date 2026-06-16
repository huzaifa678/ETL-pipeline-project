"""KFP step 1 — ingest World Bank data to S3 (no local-disk state).

Dates are dynamic: --end-year 0 means "current year", and --incremental reads the
last ingested year from an S3 state object so scheduled runs only pull new years.
Writes the raw CSV to s3://<bucket>/<raw-prefix>/raw.csv so step 2 can read it.
"""
import argparse
import logging
import os
from datetime import datetime

import boto3
import pandas as pd
import requests

logger = logging.getLogger("worldbank.steps.ingest")


def _s3():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))


def read_last_year(bucket: str, key: str, default: int) -> int:
    try:
        obj = _s3().get_object(Bucket=bucket, Key=key)
        return int(obj["Body"].read().decode().strip())
    except Exception:
        return default


def write_last_year(bucket: str, key: str, year: int) -> None:
    _s3().put_object(Bucket=bucket, Key=key, Body=str(year).encode())


def fetch(base_url, countries, indicator, start_year, end_year) -> pd.DataFrame:
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
        records.extend(data[1])
        if page >= data[0].get("pages", 1):
            break
        page += 1
    return pd.json_normalize(records)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.getenv("BASE_URL",
                   "http://api.worldbank.org/v2/countries"))
    p.add_argument("--indicator", default="SP.POP.TOTL")
    p.add_argument("--countries", default="br;cn;us;de")
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=0, help="0 = current year")
    # Value-based (not a flag) so KFP can pass it as a pipeline parameter.
    p.add_argument("--incremental", default="true",
                   type=lambda v: str(v).lower() == "true")
    p.add_argument("--bucket", required=True)
    p.add_argument("--raw-prefix", default="worldbank/raw")
    p.add_argument("--state-key", default="worldbank/state/last_ingested_year.txt")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    end_year = args.end_year if args.end_year > 0 else datetime.now().year
    start_year = args.start_year
    if args.incremental:
        start_year = read_last_year(args.bucket, args.state_key,
                                    default=args.start_year - 1) + 1

    if start_year > end_year:
        logger.info("No new data: %d > %d. Skipping.", start_year, end_year)
        return

    df = fetch(args.base_url, args.countries, args.indicator, start_year, end_year)
    if df.empty:
        logger.warning("API returned no rows for %d:%d.", start_year, end_year)
        return

    out = f"s3://{args.bucket}/{args.raw_prefix}/raw.csv"
    df.to_csv(out, index=False)  # pandas + s3fs
    logger.info("Wrote %d raw rows for %d:%d to %s",
                len(df), start_year, end_year, out)

    if args.incremental:
        write_last_year(args.bucket, args.state_key, end_year)


if __name__ == "__main__":
    main()
