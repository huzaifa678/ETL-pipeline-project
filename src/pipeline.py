import os
import logging

from src.component.model_trainer import ModelTrainer
from src.component.data_ingestion import DataIngestion
from src.component.data_transformer import DataTransformer


def run_pipeline():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ingestion = DataIngestion(
        indicator="SP.POP.TOTL",  
        countries="br;cn;us;de",   
        start_year=2000,
        end_year=2022
    )
    raw_df = ingestion.fetch_data()
    raw_file = f"{ingestion.indicator}_{ingestion.start_year}_{ingestion.end_year}.csv"
    raw_path = os.path.join(os.getcwd(), "data", "raw", raw_file)
    ingestion.save_raw(raw_df, filename=raw_file)

    transformer = DataTransformer(input_file=raw_path)
    loaded_raw = transformer.load_raw()
    clean_df = transformer.clean_data(loaded_raw)
    transformed_file = "transformed_data.csv"
    transformed_path = os.path.join(os.getcwd(), "data", "processed", transformed_file)
    transformer.save_transformed(clean_df, filename=transformed_file)

    trainer = ModelTrainer(input_file=transformed_path)
    df = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    model = trainer.train(X_train, y_train)
    trainer.evaluate(model, X_test, y_test)
    trainer.save_model(model, filename="model.pkl")

    print("\n Full ETL Pipeline Completed Successfully!")


if __name__ == "__main__":
    run_pipeline()
