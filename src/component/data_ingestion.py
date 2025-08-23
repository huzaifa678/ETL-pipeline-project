import os
import logging
from src.exception import CustomException
import pandas as pd
import requests
import sys



class DataIngestion():
    
    def __init__(self, indicator: str, countries: str , start_year: int = None, end_year: int = None):
        self.base_url = "http://api.worldbank.org/v2/countries"
        self.indicator = indicator
        self.country = countries
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = os.path.join(os.getcwd(), "data", "raw")
        os.makedirs(self.output_dir, exist_ok=True)
        
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch data from World Bank API and return as pandas DataFrame.
        Handles the pagination scenario.
        """
        params = {"format": "json", "per_page": 1000}
        if self.start_year and self.end_year:
            params["date"] = f"{self.start_year}:{self.end_year}"

        all_records = []
        page = 1
        
        url = f"{self.base_url}/{self.country}/indicator/{self.indicator}"

        while True:
            params["page"] = page
            response = requests.get(url, params=params)
            
            logging.info(f"Fetched page {page} from {url}")
            logging.info("converted the response to JSON")

            if response.status_code != 200:
                raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

            data = response.json()
            if not data or len(data) < 2:
                break

            all_records.extend(data[1])

            total_pages = data[0].get("pages", 1)
            if page >= total_pages:
                break
            page += 1

        if not all_records:
            raise Exception("No data retrieved from API")

        df = pd.json_normalize(all_records)
        logging.info("JSON normalized")
        return df

    def save_raw(self, df: pd.DataFrame, filename: str = None):
        """
        Save DataFrame as CSV in raw data folder.
        """
        try:
            if filename is None:
                filename = f"{self.indicator}_{self.start_year}_{self.end_year}.csv"
        
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            logging.info("Data has been saved in the CSV format")
            print(f"✅ Data saved at {filepath}")
            
        except Exception as e:
            CustomException(e, sys)
            
            
if __name__ == "__main__":
    ingestion = DataIngestion(
        indicator="SP.POP.TOTL",
        countries="br;cn;us;de",
        start_year=2000,
        end_year=2022
    )
    df = ingestion.fetch_data()
    ingestion.save_raw(df)
    print(df.head())
    