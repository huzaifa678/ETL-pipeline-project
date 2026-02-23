from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import os

from src.component.data_ingestion import DataIngestion
from src.component.data_transformer import DataTransformer
from src.component.model_trainer import ModelTrainer
from src.monitoring.push import push_metrics

RAW_DIR = os.path.join(os.getcwd(), "data", "raw")
PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")
MODEL_DIR = os.path.join(os.getcwd(), "model")

RAW_FILE = "SP.POP.TOTL_2000_2022.csv"
PROCESSED_FILE = "transformed_data.csv"
MODEL_FILE = "model.pkl"

default_args = {
    "owner": "airflow",
    "retries": 1,
}

def ingestion_task():
    ingestion = DataIngestion(
        indicator="SP.POP.TOTL",
        countries="br;cn;us;de",
        start_year=2000,
        end_year=2022,
    )
    df = ingestion.fetch_data()
    ingestion.save_raw(df, filename=RAW_FILE)

def transformation_task():
    input_path = os.path.join(RAW_DIR, RAW_FILE)
    transformer = DataTransformer(input_file=input_path)
    raw_df = transformer.load_raw()
    clean_df = transformer.clean_data(raw_df)
    transformer.save_transformed(clean_df, filename=PROCESSED_FILE)

def training_task():
    input_path = os.path.join(PROCESSED_DIR, PROCESSED_FILE)
    trainer = ModelTrainer(input_file=input_path)
    df = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    model, _ = trainer.train_and_log(X_train, X_test, y_train, y_test)
    trainer.evaluate(model, X_test, y_test)
    trainer.save_model(model, filename=MODEL_FILE)

def push_metrics_task():
    push_metrics(job_name="worldbank_population_etl")

with DAG(
    dag_id="worldbank_etl_ml_pipeline",
    description="ETL + ML pipeline without DVC (DVC handled in CI/CD)",
    start_date=datetime(2023, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
) as dag:

    ingest = PythonOperator(task_id="data_ingestion", python_callable=ingestion_task)
    transform = PythonOperator(task_id="data_transformation", python_callable=transformation_task)
    train = PythonOperator(task_id="model_training", python_callable=training_task)
    metrics = PythonOperator(task_id="push_prometheus_metrics", python_callable=push_metrics_task)

    ingest >> transform >> train >> metrics