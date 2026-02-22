import os

from dotenv import load_dotenv
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import socket

load_dotenv()

def push_metrics(job_name="etl_job"):
    gateway = os.getenv("PUSHGATEWAY_URL")
    if not gateway:
        print("PUSHGATEWAY_URL not set in environment")
        return

    registry = CollectorRegistry()

    g = Gauge("etl_metric", "metric for testing", registry=registry)
    g.set(1) 

    push_to_gateway(
        gateway,
        job=job_name,
        registry=registry,
        grouping_key={"instance": socket.gethostname()}
    )

    print(f"✅ Metrics pushed to {gateway}")