import logging
import os
import socket

from dotenv import load_dotenv
from prometheus_client import REGISTRY, push_to_gateway

# Importing the metrics module registers every metric (ingestion runs/failures/
# duration, rows ingested, transform duration, model RMSE/R2, ...) on the default
# REGISTRY. Pushing that registry ships whatever values the run set, not a dummy.
from src.monitoring import metrics  # noqa: F401

load_dotenv()


def push_metrics(job_name="etl_job", registry=REGISTRY):
    """Push the run's collected metrics to the Prometheus Pushgateway.

    Defaults to the global REGISTRY so all metrics set during an in-process run
    (e.g. src/pipeline.py) are shipped together.
    """
    gateway = os.getenv("PUSHGATEWAY_URL")
    if not gateway:
        logging.warning("PUSHGATEWAY_URL not set in environment")
        return

    push_to_gateway(
        gateway,
        job=job_name,
        registry=registry,
        grouping_key={"instance": socket.gethostname()},
    )

    logging.info("Metrics pushed to %s", gateway)
