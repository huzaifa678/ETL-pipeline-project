import os
import joblib
import pandas as pd


class Predictor:
    country_map = {
        "br": "Brazil",
        "cn": "China",
        "us": "United States",
        "de": "Germany"
    }

    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)

    def prepare_input(self, country: str, year: int, all_countries: list):
        full_country = self.country_map.get(country.lower())
        if not full_country:
            raise ValueError(f"Unknown country code: {country}")

        features = {"year": [year]}
        for c in all_countries[1:]:  
            features[f"country_{c}"] = [1 if full_country == c else 0]

        X_new = pd.DataFrame(features)

        if hasattr(self.model, "feature_names_in_"):
            X_new = X_new[self.model.feature_names_in_]

        return X_new

    def predict(self, country: str, year: int, all_countries: list):
        """
        Predict indicator value for a given country and year.
        """
        X_new = self.prepare_input(country, year, all_countries)
        prediction = self.model.predict(X_new)[0]
        print(f"Predicted value for {country.upper()} ({self.country_map[country.lower()]}) in {year}: {prediction:,.0f}")
        return prediction


if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), "model", "model.pkl")

    countries = ["Brazil", "China", "United States", "Germany"]

    predictor = Predictor(model_path=model_path)

    predictor.predict(country="us", year=2025, all_countries=countries)
    predictor.predict(country="cn", year=2030, all_countries=countries)