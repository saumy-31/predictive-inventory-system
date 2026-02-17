import streamlit as st
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("📦 Predictive Inventory Analytics System")

st.markdown("""
### 📊 Intelligent Retail Inventory Planning System
This system forecasts store-level demand using SARIMA and calculates safety stock 
and reorder points based on service level and lead time.
""")
st.markdown("---")


# -----------------------------
# SAFE DATA LOADING
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_path = os.path.join(BASE_DIR, "data", "train.csv")
store_path = os.path.join(BASE_DIR, "data", "store.csv")

try:
    train = pd.read_csv(train_path)
    store = pd.read_csv(store_path)
    st.success("✅ Data loaded successfully")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -----------------------------
# PREPROCESSING
# -----------------------------
train['Date'] = pd.to_datetime(train['Date'])
train = train[train['Open'] == 1]

data = train.merge(store, on='Store', how='left')

# -----------------------------
# USER INPUTS
# -----------------------------
store_id = st.selectbox("Select Store ID", sorted(data['Store'].unique()))
lead_time = st.slider("Lead Time (days)", 1, 30, 7)
service_level = st.selectbox("Service Level", [0.90, 0.95, 0.99])

z_values = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
z = z_values[service_level]

# -----------------------------
# RUN MODEL BUTTON
# -----------------------------

@st.cache_data
def get_store_data(data, store_id):
   daily_sales = get_store_data(data, store_id)
   return daily_sales


if st.button("Run Prediction"):

    store_data = data[data['Store'] == store_id]
    store_data = store_data.sort_values('Date')

    daily_sales = store_data.groupby('Date')['Sales'].sum()

    try:
        model = SARIMAX(
            daily_sales,
            order=(1,1,1),
            seasonal_order=(1,1,1,7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=30)
        forecast = forecast.clip(lower=0)

        avg_demand = forecast.mean()
        std_demand = forecast.std()

        safety_stock = z * std_demand * np.sqrt(lead_time)
        reorder_point = (avg_demand * lead_time) + safety_stock

        st.subheader("📊 30-Day Demand Forecast")
        st.line_chart(forecast)

        st.subheader("📦 Inventory Optimization Metrics")

        col1, col2 = st.columns(2)
        col1.metric("Safety Stock", f"{round(safety_stock):,} units")
        col2.metric("Reorder Point", f"{round(reorder_point):,} units")

        st.markdown("---")

        current_inventory = st.number_input(
            "Enter Current Inventory Level", 
            min_value=0
        )

        if current_inventory < reorder_point:
            st.error("⚠ Inventory below Reorder Point — Reorder Immediately!")
        else:
            st.success("✅ Inventory level is sufficient.")

    except Exception as e:
        st.error(f"Model error: {e}")
