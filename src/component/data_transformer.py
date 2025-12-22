import os
import logging
import pandas as pd
from src.exception import CustomException
import sys
from src.monitoring.metrics import (
    TRANSFORMATION_DURATION,
    ROWS_AFTER_TRANSFORM
)
import time


class DataTransformer:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.output_dir = os.path.join(os.getcwd(), "data", "processed")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_raw(self) -> pd.DataFrame:
        """
        Load the raw data from CSV.
        """
        try:
            df = pd.read_csv(self.input_file)
            logging.info(f"Loaded raw data from {self.input_file}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform the raw dataframe.
        Example steps: 
        - drop unnecessary columns
        - handle nulls
        - rename columns
        - type casting
        """
        start = time.time()
        
        try:
            keep_cols = ["country.value", "date", "value"]
            df = df[keep_cols]

            df = df.rename(columns={
                "country.value": "country",
                "date": "year",
                "value": "indicator_value"
            })

            df = df.dropna(subset=["indicator_value"])

            df["year"] = df["year"].astype(int)

            df["indicator_value"] = pd.to_numeric(df["indicator_value"], errors="coerce")

            logging.info("Data cleaned and transformed")
            ROWS_AFTER_TRANSFORM.set(len(df))
            TRANSFORMATION_DURATION.observe(time.time() - start)
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def save_transformed(self, df: pd.DataFrame, filename: str = None):
        """
        Save the transformed DataFrame as CSV.
        """
        try:
            if filename is None:
                filename = "transformed_data.csv"

            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            logging.info("Transformed data saved")
            print(f"✅ Transformed data saved at {filepath}")
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    input_path = os.path.join(os.getcwd(), "data", "raw", "SP.POP.TOTL_2000_2022.csv")

    transformer = DataTransformer(input_file=input_path)
    raw_df = transformer.load_raw()
    clean_df = transformer.clean_data(raw_df)
    transformer.save_transformed(clean_df)
    print(clean_df.head())
