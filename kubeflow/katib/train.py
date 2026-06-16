"""Katib trial: train a Ridge model on the processed World Bank data.

Katib injects --alpha for each trial; we print metrics in the
`name=value` format Katib's stdout metrics collector parses. The trial that
maximizes r2 wins; on request (--register) the model is pushed to S3 in the
same layout src/store/s3_store_model.py uses, so KServe can serve it.
"""
import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import boto3
import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("worldbank.katib.train")


def load_processed(path: str) -> pd.DataFrame:
    """Read processed data from local parquet/csv or s3://."""
    if path.startswith("s3://"):
        return pd.read_parquet(path)  # pandas+s3fs/pyarrow handles s3://
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def prepare(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=["country"], drop_first=True)
    X = df.drop(columns=["indicator_value"])
    y = df["indicator_value"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def upload_model(model, bucket: str, prefix: str, metrics: dict) -> None:
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    with tempfile.TemporaryDirectory() as tmp:
        # KServe's sklearn runtime expects a file named model.joblib.
        local = os.path.join(tmp, "model.joblib")
        joblib.dump(model, local)
        s3.upload_file(local, bucket, f"{prefix}/model.joblib")
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/metadata.json",
        Body=json.dumps({
            "model_name": "worldbank_population",
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }),
    )
    logger.info("Uploaded model to s3://%s/%s/", bucket, prefix)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--data", default=os.getenv("PROCESSED_DATA",
                   "s3://etl-dvc-bucket/worldbank/processed"))
    p.add_argument("--register", action="store_true",
                   help="upload the trained model to S3")
    p.add_argument("--bucket", default="etl-dvc-bucket")
    p.add_argument("--prefix", default="models/staging/worldbank_population")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = load_processed(args.data)
    X_train, X_test, y_train, y_test = prepare(df)

    model = Ridge(alpha=args.alpha)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    # Katib's StdOut metrics collector parses these exact `name=value` tokens
    # from stdout — they are a machine contract, NOT human logging, so they must
    # stay as plain prints (no logging prefix/timestamp).
    print(f"r2={r2}")
    print(f"rmse={rmse}")

    if args.register:
        upload_model(model, args.bucket, args.prefix, {"rmse": rmse, "r2": r2})


if __name__ == "__main__":
    main()
