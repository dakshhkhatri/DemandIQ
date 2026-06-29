import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import numpy as np

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="DemandIQ",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------
# LOAD FILES
# -----------------------------------
model = joblib.load("demand_model.pkl")
explainer = shap.TreeExplainer(model)
label_encoders = joblib.load("label_encoders1.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------------
# LOAD DATA
# -----------------------------------
data = pd.read_csv("sales_data.csv")

# -----------------------------------
# DATE FEATURES
# -----------------------------------
data["Date"] = pd.to_datetime(data["Date"])

data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["Quarter"] = data["Date"].dt.quarter

# Keep original date for charts if needed
raw_data = data.copy()

data.drop("Date", axis=1, inplace=True)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🧠 DemandIQ")
st.caption(
    "Smart inventory planning powered by explainable AI."
)

# =====================================================
# INPUT FORM
# =====================================================

st.markdown("---")

st.subheader("📝 Demand Forecast Inputs")

col1, col2, col3 = st.columns(3)

with col1:

    store_id = st.selectbox(
        "Store ID",
        sorted(data["Store ID"].unique())
    )

    category = st.selectbox(
        "Category",
        sorted(data["Category"].unique())
    )

    inventory = st.number_input(
        "Inventory Level",
        min_value=0,
        value=200,
        step=10
    )

with col2:

    product_id = st.selectbox(
        "Product",
        sorted(data["Product ID"].unique())
    )

    region = st.selectbox(
        "Region",
        sorted(data["Region"].unique())
    )

    price = st.number_input(
        "Price",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

with col3:

    weather = st.selectbox(
        "Weather Condition",
        sorted(data["Weather Condition"].unique())
    )

    seasonality = st.selectbox(
        "Seasonality",
        sorted(data["Seasonality"].unique())
    )

    discount = st.number_input(
        "Discount (%)",
        min_value=0,
        max_value=100,
        value=10
    )

# -----------------------------------
# =====================================================
# DATE INPUT
# =====================================================

selected_date = st.date_input(
    "Forecast Date"
)

year = selected_date.year
month = selected_date.month
dayofweek = selected_date.weekday()
quarter = ((month - 1) // 3) + 1

# =====================================================
# EXTRA INPUTS
# =====================================================

col4, col5, col6 = st.columns(3)

with col4:
    promotion = st.selectbox(
        "Promotion Active",
        [0, 1]
    )

with col5:
    competitor_price = st.number_input(
        "Competitor Pricing",
        min_value=0.0,
        value=60.0,
        step=1.0
    )

with col6:
    epidemic = st.selectbox(
        "Epidemic",
        [0, 1]
    )

# =====================================================
# BUILD INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame({
    "Store ID": [store_id],
    "Product ID": [product_id],
    "Category": [category],
    "Region": [region],
    "Inventory Level": [inventory],
    "Price": [price],
    "Discount": [discount],
    "Weather Condition": [weather],
    "Promotion": [promotion],
    "Competitor Pricing": [competitor_price],
    "Seasonality": [seasonality],
    "Epidemic": [epidemic],
    "Year": [year],
    "Month": [month],
    "DayOfWeek": [dayofweek],
    "Quarter": [quarter]
})

raw_input = input_data.copy()

# =====================================================
# ENCODE CATEGORICAL FEATURES
# =====================================================

for col, encoder in label_encoders.items():

    if col in input_data.columns:

        input_data[col] = encoder.transform(
            input_data[col]
        )

# =====================================================
# FEATURE ORDER
# =====================================================

input_data = input_data[feature_names]

# =====================================================
# PREDICTION
# =====================================================

# =====================================================
# PREDICTION
# =====================================================

if "generated" not in st.session_state:
    st.session_state.generated = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.button(
    "🚀 Generate Demand Forecast",
    use_container_width=True
):

    st.session_state.generated = True

    st.session_state.prediction = model.predict(
        input_data
    )[0]

if not st.session_state.generated:
    st.stop()

prediction = st.session_state.prediction
# =====================================================
# FORECAST SETTINGS
# =====================================================

st.markdown("---")

forecast_days = st.slider(
    "📅 Forecast Horizon (Days)",
    min_value=1,
    max_value=30,
    value=3
)

inventory = raw_input["Inventory Level"].iloc[0]

total_forecast_demand = prediction * forecast_days

inventory_gap = inventory - total_forecast_demand

# =====================================================
# KPI DASHBOARD
# =====================================================

st.subheader("📊 Forecast Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Daily Demand",
        f"{prediction:.0f}"
    )

with col2:
    st.metric(
        "Forecast Days",
        forecast_days
    )

with col3:
    st.metric(
        "Total Demand",
        f"{total_forecast_demand:.0f}"
    )

with col4:
    st.metric(
        "Inventory Gap",
        f"{inventory_gap:.0f}"
    )

# =====================================================
# BUSINESS RECOMMENDATION
# =====================================================

st.markdown("---")

st.subheader("🚨 Inventory Risk Assessment")

if inventory_gap < 0:

    shortage = abs(inventory_gap)

    st.error(
        f"""
        ⚠️ Inventory shortage expected.

        Predicted demand exceeds inventory by
        {shortage:.0f} units.

        Recommendation:
        Restock inventory.
        """
    )

else:

    st.success(
        f"""
        ✅ Inventory sufficient.

        Estimated remaining stock:
        {inventory_gap:.0f} units.
        """
    )
# =====================================================
# SHAP EXPLAINABILITY
# =====================================================

st.markdown("---")

st.subheader("📊 Factors Influencing Demand")

with st.spinner("Generating SHAP explanation..."):

    shap_values = explainer(input_data)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values.values[0]
    })

    shap_df["Absolute Impact"] = shap_df["Impact"].abs()

    shap_df = (
        shap_df
        .sort_values(
            by="Absolute Impact",
            ascending=False
        )
        .head(10)
    )

fig_shap = px.bar(
    shap_df,
    x="Impact",
    y="Feature",
    orientation="h",
    title="Top Features Influencing Demand Prediction"
)

st.plotly_chart(
    fig_shap,
    use_container_width=True
)

top_feature = shap_df.iloc[0]["Feature"]

st.info(
    f"📌 The strongest factor influencing this prediction is **{top_feature}**."
)
# 👇 ADD IT HERE

st.subheader("🏆 Strongest Demand Drivers")

for _, row in shap_df.head(5).iterrows():

    sign = "⬆️" if row["Impact"] > 0 else "⬇️"

    st.write(
        f"{sign} {row['Feature']} : {row['Impact']:.2f}"
    )

top_feature = shap_df.iloc[0]["Feature"]

st.info(
    f"📌 The strongest factor influencing this prediction is **{top_feature}**."
)

# =====================================================
# DEMAND VS INVENTORY
# =====================================================

st.subheader("⚖️ Demand vs Inventory")

comparison_df = pd.DataFrame({
    "Metric": [
        "Inventory",
        "Forecast Demand"
    ],
    "Value": [
        inventory,
        total_forecast_demand
    ]
})

fig3 = px.bar(
    comparison_df,
    x="Metric",
    y="Value",
    title="Inventory vs Forecast Demand"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =====================================================
#st.markdown("---")

st.subheader("🏪 Product & Store Profile")

colA, colB = st.columns(2)

with colA:

    st.info(
        f"""
        **Store ID:** {raw_input['Store ID'].iloc[0]}

        **Product:** {raw_input['Product ID'].iloc[0]}

        **Category:** {raw_input['Category'].iloc[0]}

        **Region:** {raw_input['Region'].iloc[0]}
        """
    )

with colB:

    st.info(
        f"""
        **Inventory:** {raw_input['Inventory Level'].iloc[0]}

        **Price:** ₹{raw_input['Price'].iloc[0]}

        **Discount:** {raw_input['Discount'].iloc[0]}%

        **Seasonality:** {raw_input['Seasonality'].iloc[0]}
        """
    )