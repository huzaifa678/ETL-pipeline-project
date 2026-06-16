"""Create a cron-scheduled recurring run of the World Bank KFP pipeline.

This is how KFP does "cron" — recurring runs are created against the KFP API at
runtime (there is no static manifest for them). Run this once after compiling;
the pipeline then fires on the given cron with dynamic/incremental dates.

Prereqs: `pip install kfp kfp-kubernetes` and a port-forward to the KFP API:
    kubectl -n kubeflow port-forward svc/ml-pipeline 8888:8888
Then:
    KFP_ENDPOINT=http://localhost:8888 python pipelines/create_recurring_run.py
"""
import logging
import os

from kfp import Client

logger = logging.getLogger("worldbank.recurring_run")

ENDPOINT = os.getenv("KFP_ENDPOINT", "http://localhost:8888")
CRON = os.getenv("KFP_CRON", "0 * * * *")          # hourly; standard 5-field cron
PIPELINE_YAML = os.getenv("PIPELINE_YAML", "worldbank_pipeline.yaml")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = Client(host=ENDPOINT)
    experiment = client.create_experiment(name="worldbank")
    job = client.create_recurring_run(
        experiment_id=experiment.experiment_id,
        job_name="worldbank-etl-ml-hourly",
        pipeline_package_path=PIPELINE_YAML,
        cron_expression=CRON,
        # Dynamic dates: end_year 0 -> current year, incremental -> only new years.
        params={"end_year": 0, "incremental": "true"},
        max_concurrency=1,
    )
    logger.info("Created recurring run %s on cron '%s'", job.recurring_run_id, CRON)


if __name__ == "__main__":
    main()
