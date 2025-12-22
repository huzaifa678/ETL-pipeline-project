from prometheus_client import Counter, Histogram, Gauge

INGESTION_RUNS = Counter(
    "etl_ingestion_runs_total",
    "Total number of ingestion runs"
)

INGESTION_FAILURES = Counter(
    "etl_ingestion_failures_total",
    "Total number of ingestion failures"
)

INGESTION_DURATION = Histogram(
    "etl_ingestion_duration_seconds",
    "Time spent ingesting data"
)

ROWS_INGESTED = Gauge(
    "etl_rows_ingested",
    "Number of rows ingested"
)

TRANSFORMATION_DURATION = Histogram(
    "etl_transformation_duration_seconds",
    "Time spent transforming data"
)

ROWS_AFTER_TRANSFORM = Gauge(
    "etl_rows_after_transform",
    "Rows after transformation"
)

MODEL_RMSE = Gauge(
    "ml_model_rmse",
    "Model RMSE"
)

MODEL_R2 = Gauge(
    "ml_model_r2",
    "Model R2 score"
)
