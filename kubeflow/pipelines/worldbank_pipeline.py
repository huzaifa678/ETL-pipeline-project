"""Kubeflow Pipelines (KFP v2) DAG for the World Bank ETL/ML flow.

End-to-end correct: every step reads/writes S3 (no shared local disk), so the
ingest -> transform -> train dependency chain actually passes data between pods.
Steps run the scripts in kubeflow/pipelines/steps/ baked into worldbank-ml:latest.

Dates are dynamic: end_year defaults to the current year and incremental ingest
reads the last ingested year from S3 state, so a recurring run keeps data live.

Compile:  python pipelines/worldbank_pipeline.py        # -> worldbank_pipeline.yaml
Submit:   via the KFP UI, or create_recurring_run.py for a cron schedule.
"""
import logging
import os

from kfp import compiler, dsl, kubernetes

logger = logging.getLogger("worldbank.pipeline")

IMAGE = os.getenv("PIPELINE_IMAGE", "worldbank-ml:latest")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI",
                       "http://mlflow.worldbank.svc.cluster.local:5000")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL",
                            "pushgateway.monitoring.svc.cluster.local:9091")

AWS_SECRET = os.getenv("AWS_SECRET_NAME", "aws-credentials")


@dsl.container_component
def ingest(bucket: str, raw_prefix: str, state_key: str, indicator: str,
           countries: str, start_year: int, end_year: int, incremental: str):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "steps.ingest"],
        args=[
            "--bucket", bucket, "--raw-prefix", raw_prefix,
            "--state-key", state_key, "--indicator", indicator,
            "--countries", countries,
            "--start-year", start_year, "--end-year", end_year,
            "--incremental", incremental,
        ],
    )


@dsl.container_component
def transform(bucket: str, raw_prefix: str, processed_prefix: str):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "steps.transform"],
        args=["--bucket", bucket, "--raw-prefix", raw_prefix,
              "--processed-prefix", processed_prefix],
    )


@dsl.container_component
def train(bucket: str, processed_prefix: str, model_prefix: str, mlflow_uri: str):
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "steps.train"],
        args=["--bucket", bucket, "--processed-prefix", processed_prefix,
              "--model-prefix", model_prefix, "--mlflow-uri", mlflow_uri],
    )


def _with_aws(task):
    """Inject the AWS creds secret as env vars on a step."""
    return kubernetes.use_secret_as_env(
        task,
        secret_name=AWS_SECRET,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        },
    )


@dsl.pipeline(
    name="worldbank-etl-ml",
    description="World Bank ETL + train + MLflow + S3 model, end-to-end via S3",
)
def worldbank_pipeline(
    bucket: str = "etl-dvc-bucket",
    raw_prefix: str = "worldbank/raw",
    processed_prefix: str = "worldbank/processed",
    model_prefix: str = "models/staging/worldbank_population",
    state_key: str = "worldbank/state/last_ingested_year.txt",
    indicator: str = "SP.POP.TOTL",
    countries: str = "br;cn;us;de",
    start_year: int = 2000,
    end_year: int = 0,          
    incremental: str = "true", 
):
    ingest_op = _with_aws(ingest(
        bucket=bucket, raw_prefix=raw_prefix, state_key=state_key,
        indicator=indicator, countries=countries,
        start_year=start_year, end_year=end_year, incremental=incremental,
    ))
    ingest_op.set_env_variable("BASE_URL",
                               "http://api.worldbank.org/v2/countries")

    transform_op = _with_aws(transform(
        bucket=bucket, raw_prefix=raw_prefix, processed_prefix=processed_prefix,
    )).after(ingest_op)

    train_op = _with_aws(train(
        bucket=bucket, processed_prefix=processed_prefix,
        model_prefix=model_prefix, mlflow_uri=MLFLOW_URI,
    )).after(transform_op)
    train_op.set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_URI)
    train_op.set_env_variable("PUSHGATEWAY_URL", PUSHGATEWAY_URL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    compiler.Compiler().compile(
        pipeline_func=worldbank_pipeline,
        package_path="worldbank_pipeline.yaml",
    )
    logger.info("Compiled -> worldbank_pipeline.yaml")
