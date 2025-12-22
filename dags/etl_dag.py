from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
from src.component.data_ingestion import DataIngestion
from src.component.data_transformer import DataTransformer
from prometheus_client import start_http_server
import logging

PROMETHEUS_PORT = 8000

try:
    start_http_server(PROMETHEUS_PORT)
    logging.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
except OSError:
    logging.warning(f"Prometheus server already running on port {PROMETHEUS_PORT}")


default_args = {
    "owner": "airflow",
    "retries": 1,
}

with DAG(
    dag_id="worldbank_etl_pipeline",
    default_args=default_args,
    description="ETL pipeline for World Bank data",
    start_date=datetime(2023, 1, 1),
    schedule="@daily",   
    catchup=False,
) as dag:

    def ingest_task(**kwargs):
        ingestion = DataIngestion(
            indicator="SP.POP.TOTL",
            countries="br;cn;us;de",
            start_year=2000,
            end_year=2022,
        )
        df = ingestion.fetch_data()
        ingestion.save_raw(df)

    def transform_task(**kwargs):
        input_path = os.path.join(
            os.getcwd(), "data", "raw", "SP.POP.TOTL_2000_2022.csv"
        )
        transformer = DataTransformer(input_file=input_path)
        raw_df = transformer.load_raw()
        clean_df = transformer.clean_data(raw_df)
        transformer.save_transformed(clean_df)

    ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_task,
    )

    transform = PythonOperator(
        task_id="data_transformation",
        python_callable=transform_task,
    )

    ingest >> transform
