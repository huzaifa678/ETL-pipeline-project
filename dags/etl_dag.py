from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

RAW_FILE = "SP.POP.TOTL_2000_2022.csv"
PROCESSED_FILE = "transformed_data.csv"
MODEL_FILE = "model.pkl"

default_args = {
    "owner": "airflow",
    "retries": 1,
}

def ingestion_task():
    import os
    from datetime import datetime
    from src.component.data_ingestion import DataIngestion

    raw_dir = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    state_file = os.path.join(raw_dir, "last_ingested_year.txt")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            last_year = int(f.read().strip())
    else:
        last_year = 2000 
    
    current_year = datetime.now().year
    if last_year >= current_year:
        print("No new data to ingest.")
        return

    ingestion = DataIngestion(
        indicator="SP.POP.TOTL",
        countries="br;cn;us;de",
        start_year=last_year + 1,
        end_year=current_year,
    )
    df = ingestion.fetch_data()
    filename = f"SP.POP.TOTL_{last_year+1}_{current_year}.csv"
    ingestion.save_raw(df, filename=filename)

    with open(state_file, "w") as f:
        f.write(str(current_year))
    print(f"Ingested data for years {last_year+1} to {current_year}")


def transformation_task():
    import os
    from src.component.data_transformer import DataTransformer

    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    raw_file = os.path.join(os.getcwd(), "data", "raw", RAW_FILE)
    transformer = DataTransformer(input_file=raw_file)
    raw_df = transformer.load_raw()
    clean_df = transformer.clean_data(raw_df)
    transformer.save_transformed(clean_df, filename=PROCESSED_FILE)
    print("Transformation complete")


def training_task():
    import os
    from src.component.model_trainer import ModelTrainer
    from src.store.s3_store_model import upload_model_to_s3

    model_dir = os.path.join(os.getcwd(), "model")
    os.makedirs(model_dir, exist_ok=True)

    processed_file = os.path.join(os.getcwd(), "data", "processed", PROCESSED_FILE)
    trainer = ModelTrainer(input_file=processed_file)

    df = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)

    model, metrics = trainer.register_and_transition_model(X_train, X_test, y_train, y_test)

    trainer.save_model(model, filename=MODEL_FILE, stage="staging", metrics=metrics)
    print("Training task complete")


def push_metrics_task():
    from src.monitoring.push import push_metrics

    push_metrics(job_name="worldbank_population_etl")
    print("Metrics pushed")


with DAG(
    dag_id="worldbank_etl_ml_pipeline",
    description="ETL + ML pipeline with MLflow versioning and S3 storage",
    start_date=datetime(2023, 1, 1),
    schedule="@hourly",
    catchup=False,
    default_args=default_args,
) as dag:

    ingest = PythonOperator(task_id="data_ingestion", python_callable=ingestion_task)
    transform = PythonOperator(task_id="data_transformation", python_callable=transformation_task)
    train = PythonOperator(task_id="model_training", python_callable=training_task)
    metrics = PythonOperator(task_id="push_prometheus_metrics", python_callable=push_metrics_task)

    ingest >> transform >> train >> metrics