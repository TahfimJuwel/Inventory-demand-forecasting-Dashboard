# ==============================================================================
# 🧬 FEATURE_ENGINEERING.PY
# This file contains the master function for creating all features.
# ==============================================================================
import pandas as pd
import numpy as np

def create_all_features(df):
    """
    Takes a raw dataframe and applies all feature engineering steps in a logical order.
    """
    print("--- Running Master Feature Engineering Function ---")
    
    # Ensure Date is datetime and sorted correctly for time-series features
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store ID', 'Product ID', 'Date']).reset_index(drop=True)

    # --- 1. Date-Based Features ---
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['days_left_in_month'] = df['Date'].dt.days_in_month - df['day_of_month']

    # --- 2. Cyclical Features ---
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)

    # --- 3. Grouped Operations Setup ---
    grouped = df.groupby(['Store ID', 'Product ID'])

    # --- 4. Lag Features ---
    lags = [1, 2, 3, 7, 14, 30, 365]
    for lag in lags:
        df[f'sales_lag_{lag}'] = grouped['Units Sold'].shift(lag)

    # --- 5. Rolling & Expanding Window Features ---
    shifted_sales = grouped['Units Sold'].shift(1)
    windows = [7, 14, 30]
    for window in windows:
        df[f'sales_rolling_mean_{window}'] = shifted_sales.rolling(window=window, min_periods=1).mean()
        df[f'sales_rolling_std_{window}'] = shifted_sales.rolling(window=window, min_periods=1).std()
        df[f'sales_rolling_median_{window}'] = shifted_sales.rolling(window=window, min_periods=1).median()
    df['sales_ewm_14d'] = shifted_sales.ewm(span=14, adjust=False).mean()

    # --- 6. Price, Promotion & Discount Context ---
    df['price_x_discount'] = df['Price'] * (1 - df['Discount'] / 100.0)
    df['avg_price_30d'] = grouped['Price'].shift(1).rolling(30, min_periods=1).mean()
    df['price_vs_avg'] = df['Price'] / (df['avg_price_30d'] + 1e-6)
    is_promo = df['Holiday/Promotion'] == 1
    df['promo_date'] = df['Date'].where(is_promo)
    df['last_promo_date'] = grouped['promo_date'].ffill()
    df['days_since_promo'] = (df['Date'] - df['last_promo_date']).dt.days
    df = df.drop(columns=['promo_date', 'last_promo_date'])

    # --- 7. Advanced Interaction & Ratio Features ---
    df['lag_momentum_1_7'] = df['sales_lag_1'] - df['sales_lag_7']
    df['days_of_supply_7d'] = df['Inventory Level'] / (df['sales_rolling_mean_7'] + 1e-6)

    # --- 8. Granular & Cross-Entity Features (LEAKAGE CORRECTED) ---
    df['product_month_avg'] = df.groupby(['Product ID', 'month'])['Units Sold'].transform('mean')
    # Use lag_1 for cross-entity features to prevent leaking the target
    df['category_day_avg'] = df.groupby(['Category', 'Date'])['sales_lag_1'].transform('mean')
    network_avg = df.groupby('Date')['sales_lag_1'].transform('mean')
    df['item_vs_network_avg'] = df['sales_lag_1'] - network_avg

    # --- 9. Final Cleanup ---
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    
    return df