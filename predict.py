"""
predict.py
----------
Small helper module used by the Streamlit app to load the saved
pipeline and produce predictions with basic input validation.
"""

import joblib
import pandas as pd

MODEL_PATH = "model/house_price_model.pkl"


def load_model(path: str = MODEL_PATH) -> dict:
    """Loads the saved dict containing the pipeline + metadata."""
    return joblib.load(path)


def validate_inputs(area, bedrooms, bathrooms, floors, parking, age, location, valid_locations):
    errors = []
    if area is None or area <= 0:
        errors.append("Area must be greater than 0.")
    if bedrooms is None or bedrooms <= 0:
        errors.append("Bedrooms must be at least 1.")
    if bathrooms is None or bathrooms < 0:
        errors.append("Bathrooms cannot be negative.")
    if floors is None or floors <= 0:
        errors.append("Floors must be at least 1.")
    if parking is None or parking < 0:
        errors.append("Parking cannot be negative.")
    if age is None or age < 0:
        errors.append("Age cannot be negative.")
    if location not in valid_locations:
        errors.append(f"Location must be one of: {', '.join(valid_locations)}.")
    return errors


def predict_price(model_bundle: dict, area, bedrooms, bathrooms, floors, parking, age, location) -> float:
    pipeline = model_bundle["pipeline"]
    input_df = pd.DataFrame(
        [{
            "Area": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "Parking": parking,
            "Age": age,
            "Location": location,
        }]
    )
    prediction = pipeline.predict(input_df)[0]
    return float(max(prediction, 0))
