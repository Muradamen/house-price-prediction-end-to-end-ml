import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 🎨 Page Config
# -----------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="wide")

# -----------------------------
# 🎨 Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #00c6ff;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stNumberInput, .stSlider {
        background-color: #262730;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# 📦 Load Model
# -----------------------------
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# 🏡 Header
# -----------------------------
st.title("🏡 House Price Prediction System")
st.markdown("### 🔥 Advanced ML Model (XGBoost + Feature Engineering)")
st.markdown("---")

# -----------------------------
# 🧾 Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Property Details")

    overall_qual = st.slider("Overall Quality", 1, 10, 7)
    overall_cond = st.slider("Overall Condition", 1, 10, 5)

    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
    total_bsmt_sf = st.number_input("Basement Area", 0, 3000, 1000)

    first_flr_sf = st.number_input("1st Floor Area", 500, 3000, 1000)
    second_flr_sf = st.number_input("2nd Floor Area", 0, 2000, 500)

    full_bath = st.slider("Full Bathrooms", 0, 3, 2)
    half_bath = st.slider("Half Bathrooms", 0, 2, 1)

with col2:
    st.subheader("📅 Year & Exterior")

    year_built = st.number_input("Year Built", 1900, 2023, 2000)
    year_remod = st.number_input("Year Remodeled", 1900, 2023, 2005)
    yr_sold = st.number_input("Year Sold", 2006, 2010, 2008)

    lot_area = st.number_input("Lot Area", 1000, 200000, 10000)

    garage_area = st.number_input("Garage Area", 0, 1000, 400)
    fireplaces = st.slider("Fireplaces", 0, 3, 1)

    porch = st.number_input("Open Porch Area", 0, 500, 50)
    deck = st.number_input("Wood Deck Area", 0, 500, 100)

# -----------------------------
# 🔘 Predict Button
# -----------------------------
st.markdown("---")
predict_btn = st.button("🚀 Predict Price")

# -----------------------------
# 🧠 Prediction Logic
# -----------------------------
if predict_btn:

    # Feature Engineering
    total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
    house_age = yr_sold - year_built
    total_bathrooms = full_bath + 0.5 * half_bath
    total_porch = porch + deck

    # Create DataFrame
    input_data = pd.DataFrame(
        [
            {
                "OverallQual": overall_qual,
                "OverallCond": overall_cond,
                "GrLivArea": gr_liv_area,
                "TotalBsmtSF": total_bsmt_sf,
                "1stFlrSF": first_flr_sf,
                "2ndFlrSF": second_flr_sf,
                "FullBath": full_bath,
                "HalfBath": half_bath,
                "YearBuilt": year_built,
                "YearRemodAdd": year_remod,
                "YrSold": yr_sold,
                "LotArea": lot_area,
                "GarageArea": garage_area,
                "Fireplaces": fireplaces,
                "TotalSF": total_sf,
                "HouseAge": house_age,
                "TotalBathrooms": total_bathrooms,
                "TotalPorchSF": total_porch,
            }
        ]
    )

    # Fill missing columns (IMPORTANT)
    for i in range(scaler.n_features_in_ - input_data.shape[1]):
        input_data[f"missing_{i}"] = 0

    # Align shape
    input_data = input_data.iloc[:, : scaler.n_features_in_]

    # Scale
    scaled = scaler.transform(input_data)

    # Predict (log → real)
    log_pred = model.predict(scaled)[0]
    prediction = np.expm1(log_pred)

    # -----------------------------
    # 🎉 Output
    # -----------------------------
    st.success(f"💰 Estimated Price: ${prediction:,.0f}")

    st.metric("Predicted Value", f"${prediction:,.0f}")

    st.balloons()
