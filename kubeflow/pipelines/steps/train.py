"""KFP step 3 — read processed parquet from S3, train, log to MLflow, push model.

Reads s3://<bucket>/<processed-prefix>, trains a LinearRegression (matching the
existing project model), logs params/metrics to the in-cluster MLflow, and uploads
model.joblib + metadata.json to s3://<bucket>/<model-prefix> so the KServe
InferenceService can serve it.
"""
import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import boto3
import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("worldbank.steps.train")


def prepare(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=["country"], drop_first=True)
    X = df.drop(columns=["indicator_value"])
    y = df["indicator_value"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def push_observability(metrics: dict) -> None:
    """Push model metrics to the Prometheus Pushgateway, if configured."""
    gateway = os.getenv("PUSHGATEWAY_URL")
    if not gateway:
        return
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    Gauge("ml_model_rmse", "Model RMSE", registry=registry).set(metrics["rmse"])
    Gauge("ml_model_r2", "Model R2 score", registry=registry).set(metrics["r2"])
    push_to_gateway(gateway, job="worldbank_train", registry=registry)
    logger.info("Pushed model metrics to %s", gateway)


def upload(model, bucket: str, prefix: str, metrics: dict) -> None:
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
    p.add_argument("--bucket", required=True)
    p.add_argument("--processed-prefix", default="worldbank/processed")
    p.add_argument("--model-prefix", default="models/staging/worldbank_population")
    p.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = pd.read_parquet(f"s3://{args.bucket}/{args.processed_prefix}")
    X_train, X_test, y_train, y_test = prepare(df)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }
    logger.info("metrics: %s", metrics)

    # Critical path first: ship metrics + the served model. These must not be
    # blocked by MLflow being unreachable/misconfigured.
    push_observability(metrics)
    upload(model, args.bucket, args.model_prefix, metrics)

    # MLflow tracking is best-effort — a logging failure shouldn't fail the run.
    if args.mlflow_uri:
        try:
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("ETL_Project_kfp")
            with mlflow.start_run():
                mlflow.log_param("model_type", "LinearRegression")
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as e:
            logger.warning("MLflow logging skipped: %s", e)


if __name__ == "__main__":
    main()
