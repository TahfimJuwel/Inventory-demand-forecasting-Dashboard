# 📦 Live Inventory & Sales Forecasting System

## 🌟 Overview
This project is an advanced, end-to-end **Retail Store Inventory Dashboard** built with Streamlit. It integrates a machine learning demand forecasting pipeline that predicts future sales and inventory requirements based on historical data, weather conditions, seasonality, and promotional events. 

By combining real-time inventory tracking with predictive analytics, this system helps retail managers prevent stockouts, eliminate overstocking, and make data-driven supply chain decisions.

## ✨ Key Features
- **📊 Real-Time Inventory Tracking**: Monitor opening stock, daily sales, and restocks dynamically.
- **🤖 Predictive Demand Forecasting**: Utilizes a machine learning pipeline (evaluating XGBoost, LightGBM, Regularized Regression, etc.) to predict the exact number of units to sell the next day.
- **🧠 Automated Supply Chain Metrics**: Automatically calculates the **7-Day Reorder Point (ROP)** and **Economic Order Quantity (EOQ)** to instruct when and how much to reorder.
- **📈 Advanced Feature Engineering**: Robust underlying logic that auto-generates time-series features (lags, rolling averages, seasonality indicators).
- **💻 Interactive UI**: A sleek Streamlit interface that updates daily records and handles user inputs for new operations.

## 🗂️ Project Structure
- `app.py`: The main Streamlit dashboard application for user interaction and metrics display.
- `train_pipeline.py`: A comprehensive ML pipeline that:
  - Cleans data and handles outliers natively.
  - Engages advanced feature selection via LightGBM.
  - Trains and evaluates multiple regression models (RMSE, MAE).
  - Selects and saves the Champion Demand Forecast model.
- `feature_engineering.py`: Contains the master function to create essential time-series ML features (date-based, cyclical, laps, rolling bounds).
- `retail_store_inventory.csv`: The core dataset logging daily inventory, sales, pricing, and external drivers.
- `demand_forecast_model.pkl` & `model_columns.pkl`: The serialized pre-trained Champion Model and its expected schema, ensuring smooth live-system inference.

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost lightgbm catboost plotly tensorflow matplotlib seaborn
   ```
   *(Consider creating a virtual environment via `python -m venv venv` and activating it before installing requirements)*

3. **Train the ML Model (Optional, if you wish to retrain)**:
   ```bash
   python train_pipeline.py
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 🚀 How to use
1. Launch the app and select your target `Store ID` and `Product ID` from the sidebar.
2. Under "Today's Inventory Operations", log the final units sold and restocked.
3. Configure the conditions for the next day (Weather, Seasonality, Promotions).
4. Click **"Finalize Day & Save for Tomorrow"**. The system will append your data and predict the necessary restock numbers and sales demand!
