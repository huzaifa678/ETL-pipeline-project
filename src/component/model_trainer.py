import os
import logging
import sys
import mlflow
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.exception import CustomException
from src.monitoring.metrics import MODEL_R2, MODEL_RMSE
from src.store.s3_store_model import upload_model_to_s3  

mlflow.set_tracking_uri("http://localhost:5000")

EXPERIMENT_NAME = "ETL_Project_v2" 
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)


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

    def train_and_log(self, X_train, X_test, y_train, y_test):
        with mlflow.start_run():
            model = LinearRegression()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            r2 = r2_score(y_test, preds)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("train_size", len(X_train))
            mlflow.sklearn.log_model(model, "model")

            return model, {"rmse": rmse, "r2": r2}

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