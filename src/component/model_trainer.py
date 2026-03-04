import os
import logging
from pyexpat import model
import mlflow
from mlflow import artifacts, run
from opentelemetry import metrics
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.exception import CustomException
from src.monitoring.metrics import MODEL_R2, MODEL_RMSE
from src.store.s3_store_model import upload_model_to_s3  

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


EXPERIMENT_NAME = "ETL_Project_v2" 
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

MODEL_NAME = "worldbank_population"


class ModelTrainer:
    def __init__(self, input_file: str, s3_bucket: str = "etl-dvc-bucket"):
        self.input_file = input_file
        self.model_dir = os.path.join(os.getcwd(), "model")
        os.makedirs(self.model_dir, exist_ok=True)
        self.s3_bucket = s3_bucket

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.input_file)
            logging.info(f"Loaded processed data from {self.input_file}")
            return df
        except Exception as e:
            raise CustomException(e)

    def prepare_data(self, df: pd.DataFrame):
        try:
            df = pd.get_dummies(df, columns=["country"], drop_first=True)
            X = df.drop(columns=["indicator_value"])
            y = df["indicator_value"]
            logging.info("Prepared features and target")
            return train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            raise CustomException(e)

    def train_and_log(self, X_train, X_test, y_train, y_test, run):
        if run is None:
            raise ValueError("MLflow run must be provided for logging")

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("train_size", len(X_train))
        mlflow.sklearn.log_model(model, artifact_path="models")

        run_id = run.info.run_id
        return model, {"rmse": rmse, "r2": r2}, run_id

    def evaluate(self, model, X_test, y_test):
        try:
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            r2 = r2_score(y_test, preds)
            
            MODEL_RMSE.set(rmse)
            MODEL_R2.set(r2)

            logging.info(f"Model Evaluation -> RMSE: {rmse}, R2: {r2}")
            print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")
            return {"rmse": rmse, "r2": r2}
        except Exception as e:
            raise CustomException(e)
        

    def register_and_transition_model(self, X_train, X_test, y_train, y_test, model_name="worldbank_population"):
        """
        Train, evaluate, log metrics, and register model in MLflow atomically.
        """
        client = mlflow.MlflowClient()

        try:
            client.get_registered_model(model_name)
        except mlflow.exceptions.RestException:
            client.create_registered_model(model_name)

        with mlflow.start_run() as run:
            model = LinearRegression()
            model.fit(X_train, y_train)

            metrics = self.evaluate(model, X_test, y_test)

            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("r2", metrics["r2"])
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)

            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"

            last_versions = client.get_latest_versions(model_name, stages=["Staging", "Production"])
            if last_versions:
                last_rmses = [
                    float(client.get_metric_history(v.run_id, "rmse")[-1].value) for v in last_versions
                ]
                best_rmse = min(last_rmses)
                if metrics["rmse"] >= best_rmse:
                    print(f"RMSE ({metrics['rmse']:.4f}) did not improve ({best_rmse:.4f}). Skipping registration.")
                    return model, metrics

            mv = mlflow.register_model(model_uri=model_uri, name=model_name)
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=True
            )
            print(f"Registered version {mv.version} of '{model_name}' in Staging.")

        return model, metrics

    def save_model(self, model, filename: str = "model.pkl", stage: str = "staging", metrics: dict = None):
        """
        Save the trained model locally and upload to S3.
        """
        try:
            # Save locally
            filepath = os.path.join(self.model_dir, filename)
            joblib.dump(model, filepath)
            logging.info(f"Model saved at {filepath}")
            print(f"✅ Model saved locally at {filepath}")

            if metrics is None:
                metrics = {"rmse": None, "r2": None}

            upload_model_to_s3(
                model=model,
                bucket=self.s3_bucket,
                stage=stage,
                metrics=metrics,
                model_name="worldbank_population"
            )
        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    input_path = os.path.join(os.getcwd(), "data", "processed", "transformed_data.csv")
    trainer = ModelTrainer(input_file=input_path)

    df = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)

    model, metrics = trainer.train_and_log(X_train, X_test, y_train, y_test)

    metrics = trainer.evaluate(model, X_test, y_test)

    trainer.save_model(model, stage="staging", metrics=metrics)