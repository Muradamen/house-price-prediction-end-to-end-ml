import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 🎨 Page Config
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# -----------------------------
# 🎨 Custom Styling
# -----------------------------
st.markdown("""
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
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 📦 Load Artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/xgboost_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_artifacts()

# -----------------------------
# 📊 Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

df = load_data()

# -----------------------------
# 📌 Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏡 Prediction", "📊 EDA Dashboard"]
)

# =============================
# 🏡 PREDICTION PAGE
# =============================
if page == "🏡 Prediction":

    st.title("🏡 House Price Prediction System")
    st.markdown("### 🔥 XGBoost + Feature Engineering + SHAP")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏠 Property Details")

        overall_qual = st.slider("Overall Quality", 1, 10, 7)
        overall_cond = st.slider("Overall Condition", 1, 10, 5)

        gr_liv_area = st.number_input("Living Area", 500, 5000, 1500)
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

        porch = st.number_input("Open Porch", 0, 500, 50)
        deck = st.number_input("Wood Deck", 0, 500, 100)

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Price")

    if predict_btn:

        # -----------------------------
        # 🧠 Feature Engineering
        # -----------------------------
        total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
        house_age = yr_sold - year_built
        total_bathrooms = full_bath + 0.5 * half_bath
        total_porch = porch + deck

        input_data = pd.DataFrame([{
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
            "TotalPorchSF": total_porch
        }])

        # 🔥 Align features
        input_data = input_data.reindex(columns=columns, fill_value=0)

        # Scale
        scaled = scaler.transform(input_data)

        # Predict
        log_pred = model.predict(scaled)[0]
        prediction = np.expm1(log_pred)

        # -----------------------------
        # 🎉 Output
        # -----------------------------
        st.success(f"💰 Estimated Price: ${prediction:,.0f}")
        st.metric("Predicted Value", f"${prediction:,.0f}")

        # -----------------------------
        # 🧠 Human Explanation
        # -----------------------------
        st.markdown("## 🧠 Why this price?")

        explanation = []

        if overall_qual >= 8:
            explanation.append("🏆 High quality increased price")
        if gr_liv_area > 2000:
            explanation.append("📐 Large area increased value")
        if house_age < 10:
            explanation.append("🆕 New house increased price")
        if garage_area > 500:
            explanation.append("🚗 Large garage added value")

        if not explanation:
            explanation.append("📊 Balanced features determine price")

        for e in explanation:
            st.write("✔️", e)

        # -----------------------------
        # 🔍 SHAP Explanation
        # -----------------------------
        st.markdown("## 🔍 Model Explanation (SHAP)")

        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)

        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

# =============================
# 📊 EDA DASHBOARD
# =============================
elif page == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # 📈 Price Distribution
    # -----------------------------
    st.subheader("🏷️ Sale Price Distribution")

    fig, ax = plt.subplots()
    df["SalePrice"].hist(bins=50, ax=ax)
    st.pyplot(fig)

    # -----------------------------
    # 🔥 Correlation Heatmap
    # -----------------------------
    st.subheader("🔥 Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, ax=ax2)
    st.pyplot(fig2)

    # -----------------------------
    # 🏆 Top Features
    # -----------------------------
    st.subheader("🏆 Top Features")

    top_corr = corr["SalePrice"].abs().sort_values(ascending=False)[1:10]
    st.bar_chart(top_corr)