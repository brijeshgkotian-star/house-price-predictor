"""
app.py
------
House Price Predictor — Streamlit web application.

A polished, production-style UI on top of the trained regression
pipeline. Loads the saved model once, validates user input, and
displays a clearly-labelled price estimate along with model context.
"""

import json
import os

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_price, validate_inputs

MODEL_PATH = "model/house_price_model.pkl"
METRICS_PATH = "model/metrics.json"
DATA_PATH = "data/housing.csv"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
        }

        #MainMenu, footer {visibility: hidden;}

        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #14532d 100%);
            border-radius: 18px;
            padding: 2.6rem 2.4rem;
            margin-bottom: 1.6rem;
            color: #f8fafc;
        }
        .hero h1 {
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.02em;
        }
        .hero p {
            font-size: 1.02rem;
            color: #cbd5e1;
            margin: 0;
            max-width: 680px;
            line-height: 1.5;
        }
        .badge-row { margin-top: 1rem; }
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.18);
            color: #e2e8f0;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            margin-right: 0.5rem;
            font-weight: 500;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e5e9f0;
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }

        .result-card {
            background: linear-gradient(135deg, #14532d 0%, #166534 100%);
            border-radius: 16px;
            padding: 1.8rem 2rem;
            color: white;
            text-align: center;
        }
        .result-card .label {
            font-size: 0.85rem;
            color: #bbf7d0;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .result-card .price {
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0.35rem 0;
        }
        .result-card .sub {
            font-size: 0.85rem;
            color: #dcfce7;
        }

        .disclaimer {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 0.82rem;
            color: #9a3412;
            margin-top: 1rem;
        }

        .metric-box {
            background: #f8fafc;
            border: 1px solid #e5e9f0;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            text-align: center;
        }
        .metric-box .v { font-size: 1.4rem; font-weight: 800; color: #0f172a; }
        .metric-box .k { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

        div.stButton > button {
            background: linear-gradient(135deg, #166534 0%, #14532d 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.65rem 1.2rem;
            font-weight: 700;
            width: 100%;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #14532d 0%, #0f3d20 100%);
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_inr(amount: float) -> str:
    """Formats a number into Indian currency style with Lakh/Crore label."""
    amount = round(amount)
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Crore"
    if amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.2f} Lakh"
    return f"₹{amount:,.0f}"


@st.cache_resource
def get_model_bundle():
    return load_model(MODEL_PATH)


@st.cache_data
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def get_sample_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏠 House Price Predictor</h1>
        <p>An AI-powered regression tool that estimates property prices from area,
        rooms, parking, age and location — trained and evaluated on historical housing data.</p>
        <div class="badge-row">
            <span class="badge">Supervised Learning</span>
            <span class="badge">Regression</span>
            <span class="badge">Scikit-learn</span>
            <span class="badge">Streamlit</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Load model (with graceful error handling)
# --------------------------------------------------------------------------
model_load_error = None
model_bundle = None
try:
    model_bundle = get_model_bundle()
except FileNotFoundError:
    model_load_error = (
        "No trained model found at `model/house_price_model.pkl`. "
        "Run `python src/train.py` first to train and save the model."
    )
except Exception as e:  # noqa: BLE001
    model_load_error = f"Could not load the model: {e}"

if model_load_error:
    st.error(model_load_error)
    st.stop()

metrics = get_metrics()
valid_locations = model_bundle.get("locations", [])

# --------------------------------------------------------------------------
# Sidebar: model info
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.markdown(f"**Active model:** {model_bundle.get('best_model_name', 'N/A')}")
    if metrics:
        best = metrics["results"][metrics["best_model"]]
        st.metric("R² Score", f"{best['R2']:.3f}")
        st.metric("MAE", format_inr(best["MAE"]))
        st.metric("RMSE", format_inr(best["RMSE"]))
        st.caption(f"Evaluated on a held-out 20% test split · {metrics['n_rows_after_cleaning']} rows after cleaning")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This app estimates prices using patterns learned from historical data. "
        "It is an educational ML project, not a certified property valuation."
    )
    st.markdown("---")
    st.markdown("Built with Python · Pandas · Scikit-learn · Streamlit")

# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Enter Property Details")

    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Area (sq.ft)", min_value=100, max_value=20000, value=1500, step=50)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        floors = st.number_input("Floors", min_value=1, max_value=10, value=1, step=1)
    with c2:
        parking = st.number_input("Parking spaces", min_value=0, max_value=10, value=1, step=1)
        age = st.number_input("Property age (years)", min_value=0, max_value=100, value=5, step=1)
        location = st.selectbox("Location", options=valid_locations)

    predict_clicked = st.button("🔍 Predict Price", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    if predict_clicked:
        errors = validate_inputs(
            area, bedrooms, bathrooms, floors, parking, age, location, valid_locations
        )
        if errors:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            for err in errors:
                st.error(err)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            try:
                price = predict_price(
                    model_bundle, area, bedrooms, bathrooms, floors, parking, age, location
                )
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="label">Estimated Price</div>
                        <div class="price">{format_inr(price)}</div>
                        <div class="sub">≈ ₹{price:,.0f} · based on {model_bundle.get('best_model_name')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div class="disclaimer">
                        ⚠️ This is a machine-learning estimate based on historical data patterns,
                        not a certified property valuation. Actual market prices depend on many
                        additional factors (condition, exact locality, amenities, market timing).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Result")
        st.caption("Fill in the property details and click **Predict Price** to see an estimate here.")
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Model comparison section
# --------------------------------------------------------------------------
if metrics:
    st.markdown("### 📈 Model Comparison")
    st.caption("All candidate models were trained and evaluated on the same 80/20 train-test split.")

    results_df = pd.DataFrame(metrics["results"]).T
    results_df = results_df[["MAE", "RMSE", "R2", "CV_R2_mean"]]
    results_df.index.name = "Model"
    results_df = results_df.sort_values("R2", ascending=False)

    cols = st.columns(4)
    labels = ["MAE", "RMSE", "R²", "5-Fold CV R²"]
    best_row = results_df.iloc[0]
    values = [format_inr(best_row["MAE"]), format_inr(best_row["RMSE"]), f"{best_row['R2']:.3f}", f"{best_row['CV_R2_mean']:.3f}"]
    for col, label, value in zip(cols, labels, values):
        with col:
            st.markdown(
                f"""<div class="metric-box"><div class="v">{value}</div><div class="k">{label} (best model)</div></div>""",
                unsafe_allow_html=True,
            )

    st.dataframe(
        results_df.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.3f}", "CV_R2_mean": "{:.3f}"}),
        use_container_width=True,
    )

st.markdown("---")
st.caption("House Price Predictor · AI/ML Portfolio Project · Built for educational purposes.")
