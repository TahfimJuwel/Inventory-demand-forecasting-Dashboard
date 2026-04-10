import os
import pandas as pd
import numpy as np

# ==============================
# 📊 Visualization Libraries
# ==============================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set Plotly renderers (choose one)
pio.renderers.default = "notebook_connected"
# pio.renderers.default = "iframe"

# ==============================
# ⚠️ Warnings
# ==============================
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 🤖 Machine Learning Libraries
# ==============================
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR  
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# XGBoost
from xgboost import XGBRegressor

# LightGBM
import lightgbm as lgb

# CatBoost
from catboost import CatBoostRegressor

# ==============================
# 🔮 Deep Learning (TensorFlow / Keras)
# ==============================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout



# --- Load Data ---
df = pd.read_csv('retail_store_inventory.csv')
# --- Calculate the IQR and Upper Bound ---
Q1 = df['Units Sold'].quantile(0.25)
Q3 = df['Units Sold'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + (1.5 * IQR)

# --- Identify the number of outliers before capping ---
outliers_count_before = df[df['Units Sold'] > upper_bound].shape[0]
print(f"Found {outliers_count_before} records with 'Units Sold' above the upper bound.")

# --- Cap the outliers by OVERWRITING the 'Units Sold' column ---
# This is the direct approach. It modifies the column in place.
df['Units Sold'] = np.where(
    df['Units Sold'] > upper_bound,  # Condition: If Units Sold is an outlier...
    upper_bound,                     # ...replace it with the upper_bound value...
    df['Units Sold']                 # ...otherwise, keep the original value.
)

print(f"✅ Successfully capped {outliers_count_before} outliers directly within the 'Units Sold' column.")

# --- Verify the result by checking the new maximum value ---
print("\nDescription of the 'Units Sold' column AFTER capping:")
print(df['Units Sold'].describe())


# --- 3. FEATURE ENGINEERING (Now using the master function) ---
from feature_engineering import create_all_features # <-- IMPORT THE FUNCTION
# --- 2. APPLY MASTER FEATURE ENGINEERING ---
df_processed = create_all_features(df)

low_cardinality_cols = ['Region', 'Weather Condition', 'Seasonality']
print(f"Applying one-hot encoding to low-cardinality columns: {low_cardinality_cols}")
df_processed = pd.get_dummies(df_processed, columns=low_cardinality_cols, drop_first=True)


high_cardinality_cols = ['Store ID', 'Product ID', 'Category']
print(f"Converting high-cardinality columns to 'category' dtype for efficient model handling: {high_cardinality_cols}")
for col in high_cardinality_cols:
    df_processed[col] = df_processed[col].astype('category')

# --- Final Sorting ---
# Sort final data by date before splitting. This is crucial for time series validation.
df_processed = df_processed.sort_values('Date').reset_index(drop=True)


features_to_drop_for_analysis = [
    'Units Sold',         # Target variable
    'Demand Forecast',    # Leaky baseline
    'Date',               # Original date column
    'Competitor Pricing', # Unused
    'Units Ordered',      # Leaky
]

X_analysis = df_processed.drop(columns=features_to_drop_for_analysis)
y_analysis = df_processed['Units Sold']

categorical_cols_analysis = X_analysis.select_dtypes(include=['object', 'category']).columns
print(f"One-hot encoding for analysis: {list(categorical_cols_analysis)}")
X_analysis_encoded = pd.get_dummies(X_analysis, columns=categorical_cols_analysis, drop_first=True)


print("\nTraining a discovery model to find feature importances...")
discovery_model = lgb.LGBMRegressor(random_state=42, n_estimators=200)
discovery_model.fit(X_analysis_encoded, y_analysis)
print("Discovery model training complete.")


# --- Step 3: Extract and Visualize Feature Importances ---
feature_importances = pd.DataFrame({
    'feature': X_analysis_encoded.columns,
    'importance': discovery_model.feature_importances_
}).sort_values('importance', ascending=False)

# Display the top N features
N_TOP_FEATURES = 20
print(f"\n--- Top {N_TOP_FEATURES} Most Important Features ---")
print(feature_importances.head(N_TOP_FEATURES))

NUM_FEATURES_TO_KEEP = 12
top_features = feature_importances['feature'].head(NUM_FEATURES_TO_KEEP).tolist()

print(f"\n--- Selecting the Top {NUM_FEATURES_TO_KEEP} Features for Model Training ---")
print(top_features)


X = X_analysis_encoded[top_features]
y = y_analysis


split_point = int(len(df_processed) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# Get corresponding dates for context
train_dates = df_processed.loc[X_train.index, 'Date']
test_dates = df_processed.loc[X_test.index, 'Date']

print(f"Training data from: {train_dates.min().date()} to {train_dates.max().date()}")
print(f"Testing data from:  {test_dates.min().date()} to {test_dates.max().date()}")



standard_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=42),
    "Lasso Regression": Lasso(random_state=42),
    #"K-Nearest Neighbors": KNeighborsRegressor(),
    #"Decision Tree": DecisionTreeRegressor(random_state=42),
    #"Support Vector Regressor": SVR(),
    #"Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
    #"CatBoost": CatBoostRegressor(random_state=42, verbose=0)
}
model_results = {}
for name, model in standard_models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    ### ADDED MSE AND RMSE CALCULATION ###
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    model_results[name] = {
        'mae': mae, 
        'mse': mse,
        'rmse': rmse,
        'predictions': predictions, 
        'model_object': model
    }
    print(f"✅ {name} -> MAE: {mae:.2f} | RMSE: {rmse:.2f}")

# Special Handling for LightGBM
print("\n--- Training LightGBM with Early Stopping ---")
lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=1000, n_jobs=-1)
lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mae', callbacks=[lgb.early_stopping(100, verbose=False)])
lgbm_predictions = lgbm.predict(X_test)

### ADDED MSE AND RMSE CALCULATION ###
lgbm_mae = mean_absolute_error(y_test, lgbm_predictions)
lgbm_mse = mean_squared_error(y_test, lgbm_predictions)
lgbm_rmse = np.sqrt(lgbm_mse)

model_results["LightGBM (Optimized)"] = {
    'mae': lgbm_mae,
    'mse': lgbm_mse,
    'rmse': lgbm_rmse,
    'predictions': lgbm_predictions, 
    'model_object': lgbm
}
print(f"✅ LightGBM (Optimized) -> MAE: {lgbm_mae:.2f} | RMSE: {lgbm_rmse:.2f}")


# # BiLSTM Model
# print("\n\n--- Training Deep Learning Model (BiLSTM) ---")
# # ... (BiLSTM preprocessing code remains the same) ...
# scaler_X = MinMaxScaler(); X_scaled = scaler_X.fit_transform(X)
# scaler_y = MinMaxScaler(); y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# def create_sequences(X, y, time_steps=30):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         Xs.append(X[i:(i + time_steps)])
#         ys.append(y[i + time_steps])
#     return np.array(Xs), np.array(ys)

# TIME_STEPS = 30
# X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

# n_test = len(y_test)
# X_train_seq, X_test_seq = X_seq[:-n_test], X_seq[-n_test:]
# y_train_seq, y_test_seq = y_seq[:-n_test], y_seq[-n_test:]
# print(f"Created {len(X_train_seq)} training sequences and {len(X_test_seq)} testing sequences.")

# # --- B. Build and Train the BiLSTM Model ---
# print("Building and training the BiLSTM model...")
# bilstm_model = Sequential([
#     Bidirectional(LSTM(units=50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))),
#     Dropout(0.2),
#     Dense(units=25, activation='relu'),
#     Dense(units=1)
# ])
# bilstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
# history = bilstm_model.fit(X_train_seq, y_train_seq, epochs=25, batch_size=32, validation_data=(X_test_seq, y_test_seq), verbose=0) 



# bilstm_predictions_scaled = bilstm_model.predict(X_test_seq)
# bilstm_predictions = scaler_y.inverse_transform(bilstm_predictions_scaled)
# num_predictions = len(bilstm_predictions)
# y_test_aligned = y_test.iloc[-num_predictions:]

# ### ADDED MSE AND RMSE CALCULATION ###
# bilstm_mae = mean_absolute_error(y_test_aligned, bilstm_predictions)
# bilstm_mse = mean_squared_error(y_test_aligned, bilstm_predictions)
# bilstm_rmse = np.sqrt(bilstm_mse)

# model_results["BiLSTM"] = {
#     'mae': bilstm_mae, 
#     'mse': bilstm_mse,
#     'rmse': bilstm_rmse,
#     'predictions': bilstm_predictions, 
#     'model_object': bilstm_model
# }
# print(f"✅ BiLSTM -> MAE: {bilstm_mae:.2f} | RMSE: {bilstm_rmse:.2f}")


# ==============================
# 🏆 Final Model Selection
# ==============================
print("\n\n--- Final Model Selection (based on MAE) ---")
# We will still choose the best model based on MAE as it's most interpretable
best_model_name = min(model_results, key=lambda k: model_results[k]['mae'])
best_mae = model_results[best_model_name]['mae']
best_rmse = model_results[best_model_name]['rmse']
print(f"🏆 ULTIMATE BEST MODEL: '{best_model_name}' with an MAE of {best_mae:.2f} and RMSE of {best_rmse:.2f}")


best_model_name = min(model_results, key=lambda k: model_results[k]['mae'])
best_model_object = model_results[best_model_name]['model_object']
best_mae = model_results[best_model_name]['mae']

print(f"🏆 OVERALL BEST MODEL: '{best_model_name}' with an MAE of {best_mae:.2f}")

print(f"\nUsing the best model ('{best_model_name}') to generate the final 'Predicted_Units_Sold' column...")

# It is important to predict on the full feature set `X` to populate the entire column
all_predictions = best_model_object.predict(X)

# Ensure no negative predictions and add the column to df_processed
df_processed['Predicted_Units_Sold'] = np.maximum(0, all_predictions)



import pickle

print("\n\n--- Saving the Champion Model to a File ---")

# The 'best_model_object' variable holds your trained champion model (Lasso, LGBM, etc.)
# We will now save this object to a file using pickle.
MODEL_FILENAME = 'demand_forecast_model.pkl'
with open(MODEL_FILENAME, 'wb') as f:
    pickle.dump(best_model_object, f)

print(f"✅ --- Model Saved Successfully! ---")
print(f"The best model ('{best_model_name}') has been saved to '{MODEL_FILENAME}'.")
print("You can now use this file in your live 'app.py' dashboard.")


# Save the list of columns the model was trained on. This is our "master template".
model_columns = X.columns.tolist() # 'X' is the final feature DataFrame used for training
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)
print("✅ Model columns saved to 'model_columns.pkl'.")