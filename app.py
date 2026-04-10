import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings

# We need our master feature engineering function
from feature_engineering import create_all_features 

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

@st.cache_resource
def load_assets(model_path, columns_path):
    """Loads the pre-trained model and the list of training columns."""
    print("--- Loading pre-trained model and column template... ---")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(columns_path, 'rb') as f:
            model_columns = pickle.load(f)
        return model, model_columns
    except FileNotFoundError:
        st.error("Error: Model files not found. Please run `train_pipeline.py` first.")
        st.stop()

# === PASTE THIS CORRECTED FUNCTION in app.py ===
def get_live_data(file_path):
    """Loads and sorts the current state of the inventory data CSV."""
    df = pd.read_csv(file_path)
    # which we can then drop to ensure data quality.
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df.dropna(subset=['Date'], inplace=True) # Remove any rows with invalid dates
    return df.sort_values('Date').reset_index(drop=True)

# ==============================================================================
# --- 3. MAIN APP EXECUTION ---
# ==============================================================================

# --- Load Assets ---
MODEL_FILE = 'demand_forecast_model.pkl'
COLUMNS_FILE = 'model_columns.pkl'
DATA_FILE = 'retail_store_inventory.csv'

model, model_columns = load_assets(MODEL_FILE, COLUMNS_FILE)
df_live = get_live_data(DATA_FILE)

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="Live Inventory Dashboard")
st.title("📦 Live Inventory & Sales Forecasting System")

# --- Sidebar Navigation ---
st.sidebar.title("Product Selection")
store_id_list = sorted(df_live['Store ID'].astype(str).unique())
selected_store_id = st.sidebar.selectbox("Select a Store ID", store_id_list)
products_in_store = sorted(df_live[df_live['Store ID'] == selected_store_id]['Product ID'].astype(str).unique())
selected_product_id = st.sidebar.selectbox("Select a Product ID", products_in_store)

# --- CORRECTED: "Today" is the most recent entry for the selected item ---
product_data_today = df_live[
    (df_live['Store ID'] == selected_store_id) &
    (df_live['Product ID'] == selected_product_id)
].tail(1).iloc[0]

current_business_day = product_data_today['Date'].date()

st.header(f"Status for {selected_product_id} at {selected_store_id}")
st.subheader(f"Current Business Day: {current_business_day.strftime('%Y-%m-%d')}")
st.markdown("---")


# ==============================================================================
# --- 4. DAILY OPERATIONS & UPDATE PANEL ---
# ==============================================================================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("#### Today's Inventory Operations")
    
    opening_stock = int(product_data_today['Inventory Level'])
    st.metric("Opening Stock for Today", f"{opening_stock} Units")

    st.markdown("##### Log Today's Transactions:")
    units_sold_today = st.number_input("Enter Units Sold Today:", min_value=0, step=1, key="sold")
    units_restocked_today = st.number_input("Enter Units Restocked Today:", min_value=0, step=1, key="restocked")

    closing_stock = opening_stock - units_sold_today + units_restocked_today
    st.metric("Calculated Closing Stock for Today", f"{closing_stock} Units", help="This will be the opening stock for tomorrow.")

    # --- Expander for setting tomorrow's conditions ---
    with st.expander("Set Next Day's Conditions (Optional)"):
        next_day_price = st.number_input("Price for Next Day", value=product_data_today['Price'])
        next_day_discount = st.number_input("Discount % for Next Day", min_value=0, max_value=100, value=0)
        next_day_promo = st.checkbox("Holiday/Promotion Next Day?", value=False)
        
        weather_options = df_live['Weather Condition'].unique().tolist()
        next_day_weather = st.selectbox("Weather for Next Day", options=weather_options, index=0)
        
        # --- NEW DROPDOWN ADDED HERE ---
        seasonality_options = df_live['Seasonality'].unique().tolist()
        next_day_seasonality = st.selectbox("Seasonality for Next Day", options=seasonality_options, index=0)
        # --- END OF NEW CODE ---

    if st.button("Finalize Day & Save for Tomorrow", type="primary"):
        next_day_date = current_business_day + timedelta(days=1)
        next_day_exists = not df_live[
            (df_live['Store ID'] == selected_store_id) &
            (df_live['Product ID'] == selected_product_id) &
            (df_live['Date'].dt.date == next_day_date)
        ].empty
        
        if next_day_exists:
            st.error(f"An entry for {next_day_date.strftime('%Y-%m-%d')} already exists. Cannot finalize day again.")
        else:
            original_columns = df_live.columns.tolist()
            new_record = {
                'Date': pd.to_datetime(next_day_date),
                'Store ID': selected_store_id, 'Product ID': selected_product_id,
                'Category': product_data_today['Category'], 'Region': product_data_today['Region'],
                'Inventory Level': closing_stock,
                'Units Sold': 0,
                'Units Ordered': units_restocked_today,
                'Demand Forecast': 0.0,
                'Price': next_day_price,
                'Discount': next_day_discount,
                'Weather Condition': next_day_weather,
                'Holiday/Promotion': 1 if next_day_promo else 0,
                'Competitor Pricing': product_data_today['Competitor Pricing'],
                # --- VALUE FROM NEW DROPDOWN USED HERE ---
                'Seasonality': next_day_seasonality
            }
            # Update today's record with the actual sales
            df_live.loc[product_data_today.name, 'Units Sold'] = units_sold_today
            
            # Append the new record for tomorrow
            new_record_df = pd.DataFrame([new_record])[original_columns]
            df_to_save = pd.concat([df_live, new_record_df], ignore_index=True)
            df_to_save.to_csv(DATA_FILE, index=False)
            
            st.success(f"Day finalized! Tomorrow's opening stock set to {closing_stock} units.")
            st.info("App is refreshing...")
            st.rerun()


# ==============================================================================
# --- 5. FORECAST & RECOMMENDATIONS PANEL ---
# ==============================================================================
with col2:
    st.markdown("#### Forecast & Replenishment Recommendations")
    
    # --- Predict Sales for TOMORROW ---
    # This logic now correctly creates features for a future date
    tomorrow_date_dt = current_business_day + timedelta(days=1)
    item_history = df_live[(df_live['Store ID'] == selected_store_id) & (df_live['Product ID'] == selected_product_id)].copy()
    tomorrow_placeholder = pd.DataFrame([product_data_today.to_dict()])
    tomorrow_placeholder['Date'] = pd.to_datetime(tomorrow_date_dt)
    tomorrow_placeholder['Units Sold'] = 0 
    combined_df = pd.concat([item_history, tomorrow_placeholder], ignore_index=True)
    df_with_features = create_all_features(combined_df)
    features_tomorrow = df_with_features.tail(1)
    
    # One-hot encode and align to the master template
    all_categorical_cols = [col for col in ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality'] if col in features_tomorrow.columns]
    features_tomorrow_encoded = pd.get_dummies(features_tomorrow, columns=all_categorical_cols, drop_first=True)
    X_tomorrow_aligned = features_tomorrow_encoded.reindex(columns=model_columns, fill_value=0)
    
    predicted_sale_tomorrow = max(0, round(model.predict(X_tomorrow_aligned)[0]))
    st.metric("Predicted Sales for Tomorrow", f"{predicted_sale_tomorrow} Units")
    
    # --- Calculate ROP & EOQ based on the last 7 days of REAL sales ---
    recent_history = item_history.tail(7)
    avg_demand_7d = recent_history['Units Sold'].mean()
    std_dev_7d = recent_history['Units Sold'].std()
    if pd.isna(std_dev_7d) or std_dev_7d == 0: std_dev_7d = avg_demand_7d * 0.5 
    
    # ROP based on 7-day lead time
    rop_7d = round((avg_demand_7d * 7) + (1.65 * std_dev_7d * np.sqrt(7)))
    # EOQ based on annualized 7-day demand
    annualized_demand_7d = avg_demand_7d * 365
    holding_cost = product_data_today['Price'] * 0.25
    eoq_7d = round(np.sqrt((2 * annualized_demand_7d * 100) / (holding_cost + 1e-6))) if holding_cost > 0 else 0
    
    st.metric("7-Day Reorder Point (ROP)", f"{rop_7d} Units")
    st.metric("Recommended Bulk Order (EOQ)", f"{eoq_7d} Units")
    
    st.markdown("##### Bulk Order Alert:")
    if closing_stock <= rop_7d:
        st.warning(f"**FLAGGED:** Your calculated closing stock ({closing_stock}) is below the Reorder Point ({rop_7d}). Plan a bulk order of **{eoq_7d} units** on your next supply run.", icon="⚠️")
    else:
        st.success("Stock is above the Reorder Point. No bulk order needed yet.", icon="✅")