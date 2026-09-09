# 🏠 House Price Predictor

An end-to-end **supervised machine learning regression** project that estimates
house prices from property details (area, bedrooms, bathrooms, parking, age,
and location), served through a polished **Streamlit** web application.

> Built as a portfolio project to demonstrate the full ML workflow: data
> cleaning → EDA → preprocessing → model comparison → evaluation → deployment.

---

## About

Given property details, the app returns an **estimated price** — not a
guaranteed valuation. A user enters area, bedrooms, bathrooms, floors,
parking, age, and location, and a trained regression pipeline returns a
numerical price estimate, along with the model's performance context.

## Problem Statement

Estimating a fair house price from raw data manually requires domain
expertise and is time-consuming. This project trains a model on historical
housing data to learn the relationship between property features and price,
so a new property's price can be estimated instantly and consistently.

## Dataset

- **Source:** Synthetically generated (`src/generate_dataset.py`) using
  realistic Indian real-estate pricing logic (per-sq.ft base rates by city,
  premiums for bedrooms/bathrooms/parking, depreciation with age, and
  market noise), so it behaves like a real-world dataset — including
  injected missing values, duplicate rows, and invalid records to
  legitimately exercise the cleaning step.
- **Size:** 2,015 rows generated → **1,995 rows** after cleaning.
- **Columns:** `Area`, `Bedrooms`, `Bathrooms`, `Floors`, `Parking`, `Age`,
  `Location`, `Price` (target).
- To regenerate: `python src/generate_dataset.py`

## Technologies

| Category            | Technology                     |
|----------------------|--------------------------------|
| Language             | Python                         |
| Data handling         | Pandas, NumPy                  |
| Visualization         | Matplotlib, Seaborn            |
| Machine Learning       | Scikit-learn                   |
| Web application        | Streamlit                      |
| Model persistence      | Joblib                         |

## Methodology

1. **Load** the raw CSV dataset.
2. **Clean**: drop duplicates, remove invalid records (non-positive
   area/bedrooms/price), cap extreme price outliers, impute missing
   numeric values with the column median.
3. **EDA** (`notebooks/house_price_analysis.ipynb`): distribution plots,
   area-vs-price scatter, price-by-location boxplot, correlation heatmap.
4. **Preprocess**: `ColumnTransformer` with `StandardScaler` for numeric
   features and `OneHotEncoder` for `Location`, wrapped in a single
   scikit-learn `Pipeline` so training and inference use identical
   transformations (no data leakage).
5. **Train** four regression algorithms on an 80/20 train-test split.
6. **Evaluate** using MAE, RMSE, R², and 5-fold cross-validated R².
7. **Select** the best model by test R² and save the full pipeline with
   `joblib`.
8. **Serve** predictions through a Streamlit app with input validation.

## Models Compared

Trained and evaluated on the same 80/20 split (`random_state=42`):

| Model              | MAE (₹)     | RMSE (₹)    | R²     | 5-Fold CV R² |
|--------------------|------------:|------------:|-------:|-------------:|
| Linear Regression   | 1,542,755   | 2,161,045   | 0.9167 | 0.9024       |
| Decision Tree        | 1,106,817   | 1,627,375   | 0.9528 | 0.9448       |
| Random Forest         | 913,472    | 1,291,560   | 0.9702 | 0.9664       |
| **Gradient Boosting** | **868,811** | **1,205,682** | **0.9741** | **0.9673** |

**Best model: Gradient Boosting Regressor** — selected automatically by
`src/train.py` based on test-set R², saved to `model/house_price_model.pkl`.
Full results are also written to `model/metrics.json` and displayed live in
the app's sidebar and comparison table.

*(Re-run `python src/train.py` to reproduce these numbers, or regenerate
the dataset first for a fresh run — results will vary slightly with a new
random seed or dataset.)*

## How to Run

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/house-price-predictor.git
cd house-price-predictor
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset

```bash
python src/generate_dataset.py
```

### 3. Train the model

```bash
python src/train.py
```

This cleans the data, trains all four models, evaluates them, and saves the
best pipeline to `model/house_price_model.pkl` plus a metrics report to
`model/metrics.json`.

### 4. Launch the web app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

## Project Structure

```
house-price-predictor/
├── data/
│   └── housing.csv                  # Generated dataset
├── notebooks/
│   └── house_price_analysis.ipynb   # EDA notebook
├── src/
│   ├── generate_dataset.py          # Synthetic dataset generator
│   ├── train.py                     # Cleaning + training + evaluation
│   └── predict.py                   # Model loading + prediction helpers
├── model/
│   ├── house_price_model.pkl        # Saved best pipeline
│   └── metrics.json                 # Model comparison results
├── app.py                           # Streamlit web application
├── requirements.txt
├── .gitignore
└── README.md
```

## Input Validation

The app validates every field before predicting:
- Area must be greater than 0
- Bedrooms must be at least 1
- Bathrooms, parking, and age cannot be negative
- Floors must be at least 1
- Location must be one of the categories the model was trained on

## Real-World Applications

- Real estate: initial price estimates for property analysis
- Banking: supporting property-related valuation workflows alongside
  professional checks
- Property investment: analyzing how features relate to historical prices
- Real-estate platforms: data-driven estimates as one signal among many

## Limitations

- Trained on a synthetic dataset — real markets have far more nuance
  (exact locality, construction quality, nearby amenities, demand cycles).
- Does not account for real-time market conditions or interest rates.
- This is an educational ML application, **not** a certified or
  professional property valuation service.

## Future Improvements

- Swap in a real housing dataset (e.g., a Kaggle city-level dataset).
- Add hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`).
- Add richer location-level and amenity features.
- Deploy to Streamlit Community Cloud with CI-based retraining.
- Add monitoring for model drift as market data changes over time.

---

## Interview / Hackathon Summary

> "I built a House Price Predictor using supervised machine learning
> regression. The model learns from historical housing data — area,
> bedrooms, bathrooms, parking, age, and location — to estimate prices for
> new properties. I cleaned the data, ran exploratory analysis, built a
> `ColumnTransformer`-based preprocessing pipeline to handle numeric scaling
> and categorical encoding without leakage, then compared four regression
> algorithms (Linear Regression, Decision Tree, Random Forest, Gradient
> Boosting) using MAE, RMSE, R², and 5-fold cross-validation. Gradient
> Boosting performed best with an R² of 0.974, and I deployed it behind a
> Streamlit interface with input validation and a live model-comparison
> dashboard."

---

*Educational AI/ML portfolio project. Predictions are estimates only.*
