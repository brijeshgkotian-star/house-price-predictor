"""
train.py
--------
End-to-end training pipeline for the House Price Predictor.

Steps:
    1. Load raw data
    2. Clean data (missing values, duplicates, invalid values)
    3. Build a preprocessing pipeline (scaling + one-hot encoding)
    4. Train several regression models
    5. Evaluate with MAE, RMSE, R2
    6. Select the best model and save the full pipeline to model/house_price_model.pkl
    7. Save a metrics report to model/metrics.json (used by the README + app)

Run:
    python src/train.py
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_SEED = 42
DATA_PATH = "data/housing.csv"
MODEL_PATH = "model/house_price_model.pkl"
METRICS_PATH = "model/metrics.json"

NUMERIC_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "Parking", "Age"]
CATEGORICAL_FEATURES = ["Location"]
TARGET = "Price"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Remove clearly invalid records
    df = df[df["Area"] > 0]
    df = df[df["Bedrooms"] > 0]
    df = df[df["Price"] > 0]

    # Remove extreme price outliers (beyond 3x the 99th percentile) —
    # these are data-entry errors, not genuine luxury homes, given the
    # dataset's overall distribution.
    price_cap = df["Price"].quantile(0.99) * 1.5
    df = df[df["Price"] <= price_cap]

    # Impute missing numeric values with the column median
    for col in ["Bathrooms", "Age", "Parking"]:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    df = df.reset_index(drop=True)
    print(f"Cleaning removed {before - len(df)} rows -> {len(df)} rows remain")
    return df


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=8),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_SEED, max_depth=None, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        start = time.time()
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate(y_test, preds)

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        metrics["CV_R2_mean"] = round(cv_scores.mean(), 4)
        metrics["train_seconds"] = round(time.time() - start, 2)

        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"{name:20s} -> {metrics}")

    # Select best model by test R2
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (R2={results[best_name]['R2']})")

    joblib.dump(
        {
            "pipeline": best_pipeline,
            "best_model_name": best_name,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "locations": sorted(df["Location"].unique().tolist()),
        },
        MODEL_PATH,
    )
    print(f"Saved best pipeline to {MODEL_PATH}")

    report = {
        "best_model": best_name,
        "results": results,
        "n_rows_after_cleaning": len(df),
        "test_size": 0.2,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": TARGET,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics report to {METRICS_PATH}")


if __name__ == "__main__":
    main()
